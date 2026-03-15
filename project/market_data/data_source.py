from abc import ABC, abstractmethod


class Tick:
    """A single timestep of market data returned by a MarketDataSource."""

    __slots__ = ("price", "regime", "drift", "volatility")

    def __init__(self, price, regime, drift, volatility):
        self.price = price
        self.regime = regime
        self.drift = drift
        self.volatility = volatility

    def __repr__(self):
        return (
            f"Tick(price={self.price}, regime={self.regime!r}, "
            f"drift={self.drift:.4f}, volatility={self.volatility:.4f})"
        )


class MarketDataSource(ABC):
    """
    Abstract base for all market data providers.

    Implementations:
      - SimulatedDataSource  : wraps MarketAgent for synthetic price generation
      - (future) LiveDataSource / PaperDataSource : connect to real exchanges
    """

    @property
    @abstractmethod
    def initial_regime(self) -> str:
        """The starting regime (for reporting before any ticks are produced)."""

    @abstractmethod
    def get_next_tick(self) -> Tick:
        """
        Advance the data source by one timestep and return a Tick.
        Must be called exactly once per simulation step.
        """


class SimulatedDataSource(MarketDataSource):
    """
    Thin wrapper around MarketAgent that implements the MarketDataSource interface.

    Each call to get_next_tick() delegates to MarketAgent.update_price() and
    bundles the result with the updated regime, drift, and volatility into a Tick.
    """

    def __init__(self, market_agent):
        self._market = market_agent

    @property
    def initial_regime(self) -> str:
        return self._market.regime

    def get_next_tick(self) -> Tick:
        price = self._market.update_price()
        return Tick(
            price=price,
            regime=self._market.regime,
            drift=self._market.drift,
            volatility=self._market.volatility,
        )
