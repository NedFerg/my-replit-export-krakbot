from abc import ABC, abstractmethod
from exchange.exchange import Exchange


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

    submit_order() now returns Fill objects so the simulation can inspect
    actual execution prices and quantities (slippage-adjusted, partial fills
    supported).  The Exchange manages agent state (balance, position, PnL)
    directly during matching.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange if exchange is not None else Exchange()

    def submit_order(self, agent, order_intent, market_state) -> list:
        """Delegate to Exchange.process_order and return all fills produced."""
        return self._exchange.process_order(
            agent, order_intent.side, order_intent.price
        )

    def get_market_state(self, price, regime, drift, volatility):
        return self._exchange.get_market_state(price, regime, drift, volatility)

    def fill_resting_orders(self, price) -> list:
        return self._exchange.fill_resting_orders(price)

    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None, simulation=None) -> float:
        """
        For each asset in agent.assets, ramp agent.positions[asset] toward
        agent.target_exposures[asset] using up to max_steps increments.

        When `simulation` is provided, uses simulation.dynamic_caps for
        per-asset exposure limits (regime-adaptive).  Falls back to the
        agent's static max_long / max_short dicts when not available.

        Returns
        -------
        float — sum of transaction costs across all assets this step.
        """
        total_cost = 0.0
        max_steps  = 3

        for a in agent.assets:
            price = prices.get(a, 0.0)
            if price <= 0:
                continue

            current = agent.positions[a]

            # Dynamic caps (regime-adaptive) take priority; fall back to
            # agent's static per-asset caps when not provided.
            if simulation is not None and simulation.dynamic_caps.get(a) is not None:
                cap_long  = simulation.dynamic_caps[a]
                cap_short = -0.5 * cap_long
            else:
                cap_long  = agent.max_long.get(a,  1.0)
                cap_short = agent.max_short.get(a, -1.0)

            target = max(cap_short, min(cap_long, agent.target_exposures[a]))

            # Scale all exposures toward target vol; re-clamp so we never
            # exceed the dynamic caps that were just applied above.
            if simulation is not None:
                _vscaler = getattr(simulation, "vol_scaler", 1.0)
                target = max(cap_short, min(cap_long, target * _vscaler))

            delta  = target - current

            if abs(delta) < 0.05:
                continue

            side   = "buy" if delta > 0 else "sell"
            steps  = int(round(abs(delta) * max_steps))
            change = steps * (1.0 / max_steps)
            delta_exposure = change if delta > 0 else -change

            if microstructure_fn is not None:
                _exec_price, txn_cost = microstructure_fn(price, side, delta_exposure)
            else:
                txn_cost = 0.0

            if delta > 0:
                agent.positions[a] = min(cap_long,  current + change)
            else:
                agent.positions[a] = max(cap_short, current - change)

            agent.balance      -= txn_cost
            agent.realized_pnl -= txn_cost
            total_cost         += txn_cost

        # Keep agent.position (base class, SOL) in sync for display / featurize_state
        agent.position = agent.positions.get("SOL", 0.0)

        return total_cost

    @property
    def trade_log(self):
        return self._exchange.trade_log
