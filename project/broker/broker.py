from abc import ABC, abstractmethod
from exchange.exchange import Exchange


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
        self.funding_rates     = {}   # asset → funding rate (placeholder, all 0)

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

            # --- 1. Spot target (long-only, stable) ----------------------
            spot_target = raw_target * spot_w
            spot_target = max(0.0, spot_target)              # spot cannot short
            spot_target = max(0.0, min(cap_long, spot_target))

            # --- 2. Futures target (tactical) ----------------------------
            fut_target = raw_target * fut_w

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

            # Funding-rate adjustment (placeholder; all rates are 0 in sim)
            funding = self.funding_rates.get(a, 0.0)
            if funding != 0.0:
                adj = max(0.5, min(1.5, 1.0 - 0.2 * funding))
                fut_target *= adj

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
