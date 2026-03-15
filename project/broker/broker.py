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
    def execute_exposure(self, agent, price) -> None:
        """
        For exposure-based agents: ramp agent.position toward
        agent.target_exposure and update it directly.
        No-ops for classical (non-RL) agents.
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

    def execute_exposure(self, agent, price) -> None:
        """
        Ramp agent.position (exposure float in [-1, +1]) toward
        agent.target_exposure using up to max_steps increments of 1/max_steps.

        No exchange order book is involved — the position is updated directly.
        This keeps the exposure model clean and independent of the order-book
        integer semantics used by classical agents.

        Idempotent when abs(delta) < 0.05 (dead-band to avoid churning).
        """
        delta = agent.target_exposure - agent.position
        if abs(delta) < 0.05:
            return

        max_steps = 3
        # Number of unit steps to take this call (proportional to delta)
        steps = int(round(abs(delta) * max_steps))
        exposure_change = steps * (1.0 / max_steps)

        if delta > 0:
            agent.position = min(1.0, agent.position + exposure_change)
        else:
            agent.position = max(-1.0, agent.position - exposure_change)

    @property
    def trade_log(self):
        return self._exchange.trade_log
