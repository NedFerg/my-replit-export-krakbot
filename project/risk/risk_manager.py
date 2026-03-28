class OrderIntent:
    """Lightweight struct describing a proposed order before it reaches the exchange."""

    def __init__(self, side, quantity, price):
        self.side = side
        self.quantity = quantity
        self.price = price


class RiskManager:
    """
    Enforces per-agent and global risk limits before orders reach the exchange.

    Limits checked on every order:
      - Position limit      : agent.position +/- qty must stay within [-max_position, max_position]
      - Notional limit      : qty * mid_price must not exceed max_notional_per_order
      - Per-agent drawdown  : equity drawdown from episode start must not exceed max_drawdown
      - Global drawdown     : total equity drawdown across all agents (optional kill-switch)

    ETF-strategy events (NOT risk violations):
      - etf_skips       : intentional ETF allocation skips (neutral regime, min-notional,
                          timeout recovery, cap already breached, market closed).
      - strategy_holds  : fee-hurdle or neutral-regime position holds that retain an
                          existing ETF/spot position to maximise profit.

    These are tracked separately from rejected_orders so that episode summaries
    and RL/post-run analytics can distinguish true risk violations from valid
    strategy decisions.  Use record_etf_skip() and record_strategy_hold() to
    log them; they appear in episode_summary() under their own keys.
    """

    def __init__(
        self,
        max_position=10,
        max_notional_per_order=1_000,
        max_drawdown=0.20,
        global_max_drawdown=0.30,
    ):
        self.max_position = max_position
        self.max_notional_per_order = max_notional_per_order
        self.max_drawdown = max_drawdown
        self.global_max_drawdown = global_max_drawdown

        # Per-episode state (reset by register_agents each episode)
        self._starting_equity = {}   # agent_name → float
        self._drawdown_locked = {}   # agent_name → bool
        self.rejected_orders = {}    # agent_name → list of rejection records
        self.global_kill_switch = False

        # ETF-strategy event tracking (valid strategy decisions, not risk violations).
        # Populated via record_etf_skip() / record_strategy_hold(); reset each episode.
        self.etf_skips = {}      # agent_name → list of {reason, details}
        self.strategy_holds = {} # agent_name → list of {reason, details}

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def register_agents(self, agents):
        """
        Call at the start of each episode.
        Records each agent's starting equity and clears per-episode counters.
        """
        self._starting_equity.clear()
        self._drawdown_locked.clear()
        self.rejected_orders.clear()
        self.global_kill_switch = False
        self.etf_skips.clear()
        self.strategy_holds.clear()

        for agent in agents:
            equity = agent.balance + agent.position * 0  # position is 0 at episode start
            equity = agent.balance  # balance == initial_balance after reset
            self._starting_equity[agent.name] = equity
            self._drawdown_locked[agent.name] = False
            self.rejected_orders[agent.name] = []
            self.etf_skips[agent.name] = []
            self.strategy_holds[agent.name] = []

    # ------------------------------------------------------------------
    # Per-order gate
    # ------------------------------------------------------------------

    def approve_order(self, agent, intent, market_state) -> bool:
        """
        Returns True if the order passes all risk checks; False if it should be blocked.
        Rejection reasons are recorded in self.rejected_orders[agent.name].
        """
        mid = market_state.mid_price

        def _reject(reason):
            self.rejected_orders[agent.name].append({
                "reason": reason,
                "equity": round(agent.balance + agent.position * mid, 2),
                "position": agent.position,
            })
            return False

        # Global kill-switch — no orders allowed once triggered
        if self.global_kill_switch:
            return _reject("global_kill_switch")

        # Drawdown lock — agent exceeded its drawdown limit this episode
        if self._drawdown_locked.get(agent.name, False):
            return _reject("drawdown_lock")

        # ---- Position limit ----
        if intent.side == "buy":
            new_pos = agent.position + intent.quantity
            if new_pos > self.max_position:
                return _reject(
                    f"position_limit (buy would reach {new_pos} > {self.max_position})"
                )
        elif intent.side == "sell":
            new_pos = agent.position - intent.quantity
            if new_pos < -self.max_position:
                return _reject(
                    f"position_limit (sell would reach {new_pos} < {-self.max_position})"
                )

        # ---- Notional limit ----
        notional = intent.quantity * mid
        if notional > self.max_notional_per_order:
            return _reject(
                f"notional_limit ({notional:.2f} > {self.max_notional_per_order})"
            )

        # ---- Per-agent drawdown limit ----
        equity = agent.balance + agent.position * mid
        starting = self._starting_equity.get(agent.name, equity)
        if starting > 0:
            drawdown = (starting - equity) / starting
            if drawdown >= self.max_drawdown:
                self._drawdown_locked[agent.name] = True
                return _reject(
                    f"drawdown_limit ({drawdown:.1%} >= {self.max_drawdown:.1%})"
                )

        return True

    # ------------------------------------------------------------------
    # Global risk check (call once per step)
    # ------------------------------------------------------------------

    def check_global_risk(self, agents):
        """
        Triggers the global kill-switch if total equity has fallen too far.
        Should be called after all per-step equity updates.
        """
        if self.global_kill_switch:
            return  # already triggered

        total_starting = sum(
            self._starting_equity.get(a.name, 0) for a in agents
        )
        if total_starting <= 0:
            return

        total_equity = sum(a.balance + a.unrealized_pnl for a in agents)
        drawdown = (total_starting - total_equity) / total_starting
        if drawdown >= self.global_max_drawdown:
            self.global_kill_switch = True

    # ------------------------------------------------------------------
    # ETF-strategy event recording (valid decisions, not risk violations)
    # ------------------------------------------------------------------

    def record_etf_skip(
        self,
        agent_name: str,
        reason: str,
        details: dict | None = None,
    ) -> None:
        """
        Record an ETF allocation skip as a valid strategy decision.

        ETF skips are NOT risk violations — they are intentional strategy
        choices such as:
          - "neutral_regime"     : no directional ETF signal; hold existing positions
          - "min_notional"       : 30% of available cash is below the dust threshold
          - "timeout_recovery"   : a stale pending ETF order was cleared
          - "cap_breached"       : combined ETF allocation already at the 30% cap
          - "market_closed"      : market hours outside the ETF trading window
          - "no_price"           : ETF price unavailable (market may be closed)

        These events appear in episode_summary() under "etf_skips_per_agent"
        so RL analytics can see them without conflating them with risk blocks.

        Parameters
        ----------
        agent_name : str   Agent name, or "ETF_LAYER" for broker-level skips.
        reason     : str   One of the reason codes listed above.
        details    : dict  Optional extra context (e.g. notional, regime dict).
        """
        if agent_name not in self.etf_skips:
            self.etf_skips[agent_name] = []
        self.etf_skips[agent_name].append({"reason": reason, "details": details or {}})

    def record_strategy_hold(
        self,
        agent_name: str,
        reason: str,
        details: dict | None = None,
    ) -> None:
        """
        Record a fee-hurdle or neutral-regime position hold as a valid strategy decision.

        Strategy holds are profit-maximising decisions to retain an existing
        ETF or spot position rather than closing or rotating it:
          - "fee_hurdle"      : expected gain from closing is less than 2.5× round-trip fee
          - "neutral_regime"  : signal is indeterminate; hold to avoid unnecessary fee drag

        These events appear in episode_summary() under "strategy_holds_per_agent"
        and are clearly distinguishable from risk blocks in both logs and summaries.

        Parameters
        ----------
        agent_name : str   Agent name, or "ETF_LAYER" for broker-level holds.
        reason     : str   "fee_hurdle" or "neutral_regime".
        details    : dict  Optional context (e.g. notional, hurdle amount, regime dir).
        """
        if agent_name not in self.strategy_holds:
            self.strategy_holds[agent_name] = []
        self.strategy_holds[agent_name].append({"reason": reason, "details": details or {}})

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def episode_summary(self):
        """
        Return a dict summarising risk events for the current episode.

        Keys
        ----
        rejected_per_agent      : {name: int} — orders blocked by a risk limit
                                  (position, notional, drawdown, kill-switch)
        drawdown_locked         : [name, …]   — agents locked out this episode
        global_kill_switch      : bool        — whether the global halt fired
        etf_skips_per_agent     : {name: int} — intentional ETF skips (valid strategy)
        strategy_holds_per_agent: {name: int} — fee-hurdle / neutral-regime holds (valid)

        The last two groups are valid strategy decisions, not errors or risk
        violations, and must never be conflated with the first three groups
        in logs, dashboards, or RL reward signals.
        """
        return {
            "rejected_per_agent": {
                name: len(events)
                for name, events in self.rejected_orders.items()
            },
            "drawdown_locked": [
                name for name, locked in self._drawdown_locked.items() if locked
            ],
            "global_kill_switch": self.global_kill_switch,
            # -- Valid strategy events (not risk violations) ----------------
            "etf_skips_per_agent": {
                name: len(events)
                for name, events in self.etf_skips.items()
                if events
            },
            "strategy_holds_per_agent": {
                name: len(events)
                for name, events in self.strategy_holds.items()
                if events
            },
        }
