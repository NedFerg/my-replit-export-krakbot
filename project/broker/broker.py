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
    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None) -> float:
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

    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None) -> float:
        """
        For each asset in agent.assets, ramp agent.positions[asset] toward
        agent.target_exposures[asset] using up to max_steps increments.

        The same dead-band (< 0.05), ramp logic, and microstructure cost
        model that the single-asset execute_exposure used are applied
        independently per asset.  agent.position is kept in sync with the
        SOL position for backward-compatible display and featurize_state().

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
            target  = agent.target_exposures[a]
            delta   = target - current

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
                agent.positions[a] = min(1.0, current + change)
            else:
                agent.positions[a] = max(-1.0, current - change)

            agent.balance      -= txn_cost
            agent.realized_pnl -= txn_cost
            total_cost         += txn_cost

        # Keep agent.position (base class, SOL) in sync for display / featurize_state
        agent.position = agent.positions.get("SOL", 0.0)

        return total_cost

    @property
    def trade_log(self):
        return self._exchange.trade_log
