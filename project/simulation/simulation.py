from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.market_agent import MarketAgent
from exchange.exchange import Exchange
from config.config import (
    SIMULATION_STEPS,
    INITIAL_BALANCE,
    MARKET_START_PRICE
)


class Simulation:
    def __init__(self):
        self.market = MarketAgent(MARKET_START_PRICE)
        self.exchange = Exchange(self.market)
        self.initial_regime = self.market.regime
        self.price_history = []
        self.regime_history = []
        self.agents = [
            ValueTrader("ValueTrader", INITIAL_BALANCE),
            MomentumTrader("MomentumTrader", INITIAL_BALANCE),
            RandomTrader("RandomTrader", INITIAL_BALANCE)
        ]

    def run(self):
        for step in range(SIMULATION_STEPS):
            # Update market price (applies regime effects internally)
            price = self.exchange.update_market()

            # Build order-book-aware market state including regime signals
            state = self.exchange.get_market_state(
                price,
                self.market.regime,
                self.market.drift,
                self.market.volatility
            )

            # Each agent decides using the full market state
            for agent in self.agents:
                action = agent.decide(state)
                self.exchange.process_order(agent, action, price)

            # Fill any unmatched orders via market maker
            self.exchange.fill_resting_orders(price)

            # Update tracking for next step
            for agent in self.agents:
                agent.update_last_price(state.mid_price)
                agent.update_unrealized_pnl(price)
                agent.record_equity()

            # Record price and regime history
            self.price_history.append(price)
            self.regime_history.append(self.market.regime)

        return self.exchange.trade_log, self.agents, self.price_history, self.regime_history
