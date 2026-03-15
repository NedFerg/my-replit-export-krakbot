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
    def execute_exposure(self, agent, price, microstructure_fn=None) -> float:
        """
        Ramp agent.position toward agent.target_exposure and update directly.
        Applies microstructure costs (spread + slippage + fee) when
        microstructure_fn is supplied.

        Parameters
        ----------
        agent            : TraderAgent subclass with IS_RL_AGENT = True
        price            : float — current mid market price
        microstructure_fn: callable(mid_price, side) → (exec_price, txn_cost)
                           or None for zero-cost mode

        Returns
        -------
        float — transaction cost charged this step (0.0 if no trade or no fn)
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

    def execute_exposure(self, agent, price, microstructure_fn=None) -> float:
        """
        Ramp agent.position (exposure float in [-1, +1]) toward
        agent.target_exposure using up to max_steps increments of 1/max_steps.

        When microstructure_fn is provided the execution price includes the
        bid/ask spread and slippage, and the flat transaction fee is deducted
        from agent.balance.  The fee is returned so the caller can subtract
        it from the RL reward signal.

        No exchange order book is involved — position is updated directly.
        Idempotent when abs(delta) < 0.05 (dead-band to avoid churning).

        Returns
        -------
        float — transaction cost charged this step (0.0 if no trade).
        """
        delta = agent.target_exposure - agent.position
        if abs(delta) < 0.05:
            return 0.0

        side = "buy" if delta > 0 else "sell"

        max_steps = 3
        steps = int(round(abs(delta) * max_steps))
        exposure_change = steps * (1.0 / max_steps)

        # Signed exposure change this step — passed into apply_microstructure
        # so the dynamic slippage model can scale with trade size.
        delta_exposure = exposure_change if delta > 0 else -exposure_change

        # Apply microstructure friction when a pricing function is available.
        # exec_price is recorded on the agent for reference; the flat txn_cost
        # is deducted from balance and returned for the reward shaping.
        if microstructure_fn is not None:
            exec_price, txn_cost = microstructure_fn(price, side, delta_exposure)
        else:
            exec_price, txn_cost = price, 0.0

        if delta > 0:
            agent.position = min(1.0, agent.position + exposure_change)
        else:
            agent.position = max(-1.0, agent.position - exposure_change)

        # Deduct the flat transaction fee from the agent's cash balance.
        # This is distinct from slippage (which affects exec_price) and
        # makes the balance decrease visibly when the agent trades.
        agent.balance  -= txn_cost
        agent.realized_pnl -= txn_cost   # fee reduces realised book too

        return txn_cost

    @property
    def trade_log(self):
        return self._exchange.trade_log
