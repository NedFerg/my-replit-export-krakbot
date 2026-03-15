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
    def submit_order(self, agent, order_intent, market_state):
        """Route an approved order intent to the underlying execution venue."""

    @abstractmethod
    def get_market_state(self, price, regime, drift, volatility):
        """Return a MarketState snapshot for the current timestep."""

    @abstractmethod
    def fill_resting_orders(self, price):
        """Settle any unmatched resting orders at end of step."""

    @property
    @abstractmethod
    def trade_log(self):
        """Read-only access to the list of filled trades this episode."""


class SimulatedBroker(Broker):
    """
    Routes orders to the in-process Exchange (order book + market maker).

    This is a thin delegation layer: every method calls through to the
    underlying Exchange instance. Future brokers (paper, live) implement
    the same interface without touching the order-book code.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange if exchange is not None else Exchange()

    def submit_order(self, agent, order_intent, market_state):
        """Delegate to Exchange.process_order using the order intent fields."""
        self._exchange.process_order(agent, order_intent.side, order_intent.price)

    def get_market_state(self, price, regime, drift, volatility):
        return self._exchange.get_market_state(price, regime, drift, volatility)

    def fill_resting_orders(self, price):
        self._exchange.fill_resting_orders(price)

    @property
    def trade_log(self):
        return self._exchange.trade_log
