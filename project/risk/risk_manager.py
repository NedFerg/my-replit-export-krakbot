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

        for agent in agents:
            equity = agent.balance + agent.position * 0  # position is 0 at episode start
            equity = agent.balance  # balance == initial_balance after reset
            self._starting_equity[agent.name] = equity
            self._drawdown_locked[agent.name] = False
            self.rejected_orders[agent.name] = []

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
    # Reporting
    # ------------------------------------------------------------------

    def episode_summary(self):
        """Return a dict summarising risk events for the current episode."""
        return {
            "rejected_per_agent": {
                name: len(events)
                for name, events in self.rejected_orders.items()
            },
            "drawdown_locked": [
                name for name, locked in self._drawdown_locked.items() if locked
            ],
            "global_kill_switch": self.global_kill_switch,
        }
