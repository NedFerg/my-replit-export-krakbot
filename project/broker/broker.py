from abc import ABC, abstractmethod
from exchange.exchange import Exchange


# --- Per-asset beta categories ----------------------------------------
# High-beta alts get more futures overlay (tactical/reactive).
# Low-beta majors get more spot core (stable/conviction).
# Mid-beta stays at the cycle-phase default.
HIGH_BETA = {"SOL", "AVAX", "LINK", "HBAR"}
MID_BETA  = {"XLM", "XRP"}
LOW_BETA  = {"BTC", "ETH"}


def _adjust_weights_for_asset(asset: str, spot_w: float, fut_w: float):
    """
    Tilt the cycle-phase base weights by asset beta, then renormalise
    so spot_w + fut_w == 1.0.

        High-beta alts  → −10 % spot, +10 % futures (more tactical)
        Low-beta majors → +10 % spot, −10 % futures (more stable)
        Mid-beta        → no change
    """
    if asset in HIGH_BETA:
        spot_w *= 0.9
        fut_w  *= 1.1
    elif asset in LOW_BETA:
        spot_w *= 1.1
        fut_w  *= 0.9
    # MID_BETA: unchanged

    total = spot_w + fut_w
    if total > 0:
        spot_w /= total
        fut_w  /= total

    return spot_w, fut_w


def _get_spot_futures_weights(cycle_phase: int):
    """
    Return (spot_weight, futures_weight) based on the long-term cycle phase.

    Cycle phases:
        0 = early accumulation → lean spot (stability)
        1 = mid expansion      → balanced
        2 = late distribution  → lean futures (nimble exit)
        3 = post-crash reset   → heaviest futures (fastest reaction)
    """
    if cycle_phase == 0:
        return 0.70, 0.30
    elif cycle_phase == 1:
        return 0.60, 0.40
    elif cycle_phase == 2:
        return 0.50, 0.50
    else:   # 3 = reset
        return 0.40, 0.60


class Broker(ABC):
    """
    Abstract execution venue for agent orders.

    All order routing, market state queries, and end-of-step settlement go
    through this interface. The Simulation never touches Exchange directly —
    only the Broker does. Swapping SimulatedBroker for a KrakenBroker or
    AlpacaBroker requires no changes to Simulation or agent code.
    """

    @abstractmethod
    def submit_order(self, agent, order_intent, market_state) -> list:
        """
        Route an approved order intent to the underlying execution venue.
        Returns a list of Fill objects produced by the execution.
        """

    @abstractmethod
    def get_market_state(self, price, regime, drift, volatility):
        """Return a MarketState snapshot for the current timestep."""

    @abstractmethod
    def fill_resting_orders(self, price) -> list:
        """Settle any unmatched resting orders at end of step.
        Returns a list of Fill objects from the final match."""

    @abstractmethod
    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None, simulation=None) -> float:
        """
        For each asset in agent.assets, ramp agent.positions[asset] toward
        agent.target_exposures[asset] and apply microstructure costs.

        Parameters
        ----------
        agent            : ReinforcementLearningTrader
        prices           : Dict[str, float] — current mid price per asset
        microstructure_fn: callable(mid_price, side, delta_exposure)
                           → (exec_price, txn_cost), or None for zero-cost mode

        Returns
        -------
        float — total transaction cost charged across all assets this step
        """

    @property
    @abstractmethod
    def trade_log(self):
        """Read-only access to the list of filled trades this episode."""


class SimulatedBroker(Broker):
    """
    Routes orders to the in-process Exchange (order book + market maker).

    Execution is split into two conceptual layers:

        Spot layer   — slow, stable, conviction exposure.  Only long.
                       Executes when the position change exceeds 2%.
        Futures layer — fast, tactical.  Both long and short allowed.
                       Executes every step.  Inherits vol_scaler,
                       panic_risk, and top_risk adjustments.

    The combined exposure (spot + futures) is written to agent.positions[a]
    so the rest of the system sees a single consistent value.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange if exchange is not None else Exchange()

        # --- Hybrid execution state ---------------------------------------
        # Lazily populated per-asset on first encounter so the broker does
        # not need to know the asset universe at construction time.
        self.spot_positions    = {}   # asset → current spot exposure
        self.futures_positions = {}   # asset → current futures exposure
        self.funding_rates     = {}   # asset → raw funding rate (feed externally or sim)

        # Funding-rate EMA smoothing — reduces noise in short funding windows
        self.funding_ema   = {}   # asset → smoothed funding rate
        self.funding_alpha = 0.2  # EMA decay (0 = no update, 1 = no smoothing)

        # --- Portfolio-level leverage limits ------------------------------
        # All values are in fractional-exposure units (1.0 = 100% of equity).
        # Spot layer is excluded — leverage control applies to futures only.
        self.max_portfolio_leverage = 2.0   # total |spot + futures| cap
        self.max_futures_notional   = 1.5   # total |futures| cap across assets
        self.max_asset_futures      = 0.40  # per-asset futures cap

    # ------------------------------------------------------------------
    # Standard order-book routing (used by non-RL agents)
    # ------------------------------------------------------------------

    def submit_order(self, agent, order_intent, market_state) -> list:
        """Delegate to Exchange.process_order and return all fills produced."""
        return self._exchange.process_order(
            agent, order_intent.side, order_intent.price
        )

    def get_market_state(self, price, regime, drift, volatility):
        return self._exchange.get_market_state(price, regime, drift, volatility)

    def fill_resting_orders(self, price) -> list:
        return self._exchange.fill_resting_orders(price)

    # ------------------------------------------------------------------
    # Hybrid portfolio execution (RL agent only)
    # ------------------------------------------------------------------

    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None, simulation=None) -> float:
        """
        Two-layer hybrid execution:

        1. Spot layer  (spot_weight × target, long-only, 2% move threshold)
           Slow and stable — represents core conviction positions.

        2. Futures layer (futures_weight × target, long/short allowed)
           Fast and tactical — inherits vol_scaler, panic_risk, top_risk.
           Updates every step (1% threshold).

        Returns
        -------
        float — sum of transaction costs across all assets this step.
        """
        total_cost = 0.0

        # Pull simulation-level signals (graceful defaults when None)
        cycle_phase = getattr(simulation, "cycle_phase", 0)
        vol_scaler  = getattr(simulation, "vol_scaler",  1.0)
        panic_risk  = getattr(simulation, "panic_risk",  0)
        top_risk    = getattr(simulation, "top_risk",    0)

        spot_w, fut_w = _get_spot_futures_weights(cycle_phase)

        for a in agent.assets:
            price = prices.get(a, 0.0)
            if price <= 0:
                continue

            # Lazy-init per-asset hybrid state
            if a not in self.spot_positions:
                self.spot_positions[a]    = 0.0
                self.futures_positions[a] = 0.0
                self.funding_rates[a]     = 0.0

            # --- Caps (regime-adaptive → agent static fallback) ----------
            if simulation is not None and simulation.dynamic_caps.get(a) is not None:
                cap_long  = simulation.dynamic_caps[a]
                cap_short = -0.5 * cap_long
            else:
                cap_long  = agent.max_long.get(a,  1.0)
                cap_short = agent.max_short.get(a, -1.0)

            # Clamped RL target (pre-split)
            raw_target = max(cap_short, min(cap_long, agent.target_exposures[a]))

            # Per-asset beta tilt on top of cycle-phase base weights
            _spot_w, _fut_w = _adjust_weights_for_asset(a, spot_w, fut_w)

            # --- 1. Spot target (long-only, stable) ----------------------
            spot_target = raw_target * _spot_w
            spot_target = max(0.0, spot_target)              # spot cannot short
            spot_target = max(0.0, min(cap_long, spot_target))

            # --- 2. Futures target (tactical) ----------------------------
            fut_target = raw_target * _fut_w

            # Volatility scaling — applied to futures only
            fut_target *= vol_scaler

            # Panic de-risking — futures collapse first
            if panic_risk == 1:
                fut_target *= 0.5
            elif panic_risk == 2:
                fut_target *= 0.1

            # Blow-off top — trim futures into strength
            if top_risk == 1:
                fut_target *= 0.7
            elif top_risk == 2:
                fut_target *= 0.3

            # --- Funding-aware adjustment (EMA-smoothed) -----------------
            # Positive funding → longs pay shorts → reduce long futures.
            # Negative funding → shorts pay longs → increase long futures.
            # k=0.25 gives a 25% reduction per unit of funding; adj clamped
            # to [0.5, 1.5] to prevent extreme reactions.
            _raw_funding = self.funding_rates.get(a, 0.0)
            self.update_funding_rate(a, _raw_funding)
            funding = self.funding_ema.get(a, 0.0)
            funding_adj = max(0.5, min(1.5, 1.0 - 0.25 * funding))
            fut_target *= funding_adj

            # Log only when funding is extreme (≥5% per 8 h is unusual)
            if abs(funding) > 0.05:
                print(f"[Funding] {a}: raw={_raw_funding:.4f}  ema={funding:.4f}  adj={funding_adj:.3f}")

            # Clamp futures within caps
            fut_target = max(cap_short, min(cap_long, fut_target))

            # --- 3. Execute spot (slow — 2% threshold) -------------------
            delta_spot = spot_target - self.spot_positions[a]
            if abs(delta_spot) > 0.02:
                spot_cost = self._execute_spot_trade(
                    a, price, delta_spot, microstructure_fn
                )
                agent.balance      -= spot_cost
                agent.realized_pnl -= spot_cost
                total_cost         += spot_cost
                self.spot_positions[a] = spot_target

            # --- 4. Execute futures (fast — 1% threshold) ----------------
            delta_fut = fut_target - self.futures_positions[a]

            # --- Portfolio-level leverage enforcement --------------------
            # Works in fractional-exposure units (equity cancels out).
            # Spot is untouched — only futures are constrained here.
            current_fut = self.futures_positions[a]

            # 1. Per-asset futures cap (40 % of equity)
            fut_target = max(-self.max_asset_futures,
                             min(self.max_asset_futures, fut_target))
            delta_fut = fut_target - current_fut

            # 2. Total futures notional cap across portfolio
            _ts, _tf, _te = self._compute_portfolio_exposure()
            if _tf + abs(delta_fut) > self.max_futures_notional:
                _allowed = max(0.0, self.max_futures_notional - _tf)
                delta_fut  = (min(delta_fut,  _allowed) if delta_fut > 0
                              else max(delta_fut, -_allowed))
                fut_target = current_fut + delta_fut

            # 3. Total portfolio leverage cap (spot + futures)
            if _te + abs(delta_fut) > self.max_portfolio_leverage:
                _allowed = max(0.0, self.max_portfolio_leverage - _te)
                delta_fut  = (min(delta_fut,  _allowed) if delta_fut > 0
                              else max(delta_fut, -_allowed))
                fut_target = current_fut + delta_fut

            if abs(delta_fut) > 0.01:
                fut_cost = self._execute_futures_trade(
                    a, price, delta_fut, microstructure_fn
                )
                agent.balance      -= fut_cost
                agent.realized_pnl -= fut_cost
                total_cost         += fut_cost
                self.futures_positions[a] = fut_target

            # --- 5. Write combined exposure back to agent ----------------
            combined = self.spot_positions[a] + self.futures_positions[a]
            # Soft clamp: combined may slightly exceed a single-layer cap
            # because each layer is independently capped, but keep it sane.
            combined = max(cap_short - abs(cap_short), min(cap_long * 1.5, combined))
            agent.positions[a] = combined

        # Keep agent.position (base class, SOL) in sync for display / featurize_state
        agent.position = agent.positions.get("SOL", 0.0)
        return total_cost

    # ------------------------------------------------------------------
    # Portfolio exposure helpers
    # ------------------------------------------------------------------

    def _compute_portfolio_exposure(self):
        """
        Return (total_spot, total_futures, total_combined) as sums of
        absolute fractional exposures across all tracked assets.
        """
        total_spot = sum(abs(v) for v in self.spot_positions.values())
        total_fut  = sum(abs(v) for v in self.futures_positions.values())
        return total_spot, total_fut, total_spot + total_fut

    # ------------------------------------------------------------------
    # Funding-rate helpers
    # ------------------------------------------------------------------

    def update_funding_rate(self, asset, raw_funding):
        """
        Feed a raw funding rate into the per-asset EMA.

        Initialises the EMA to raw_funding on the first call so there is
        no cold-start discontinuity.  Call this each step (or each funding
        period) before execute_portfolio_exposure.
        """
        if asset not in self.funding_ema:
            self.funding_ema[asset] = raw_funding
        else:
            prev = self.funding_ema[asset]
            self.funding_ema[asset] = prev + self.funding_alpha * (raw_funding - prev)

    # ------------------------------------------------------------------
    # Execution sub-routines
    # ------------------------------------------------------------------

    def _execute_spot_trade(self, asset, price, delta_exposure, microstructure_fn):
        """Apply microstructure cost for a spot exposure change and return it."""
        if microstructure_fn is None:
            return 0.0
        side = "buy" if delta_exposure > 0 else "sell"
        _exec_price, cost = microstructure_fn(price, side, delta_exposure)
        return cost

    def _execute_futures_trade(self, asset, price, delta_exposure, microstructure_fn):
        """Apply microstructure cost for a futures exposure change and return it."""
        if microstructure_fn is None:
            return 0.0
        side = "buy" if delta_exposure > 0 else "sell"
        _exec_price, cost = microstructure_fn(price, side, delta_exposure)
        return cost

    @property
    def trade_log(self):
        return self._exchange.trade_log
