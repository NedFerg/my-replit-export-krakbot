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
        self.agents = [
            ValueTrader("ValueTrader", INITIAL_BALANCE),
            MomentumTrader("MomentumTrader", INITIAL_BALANCE),
            RandomTrader("RandomTrader", INITIAL_BALANCE)
        ]

    def run(self):
        for step in range(SIMULATION_STEPS):
            # Update market price
            price = self.exchange.update_market()

            # Each agent decides and submits orders
            for agent in self.agents:
                action = agent.decide(price)
                self.exchange.process_order(agent, action, price)

            # Fill any unmatched orders via market maker
            self.exchange.fill_resting_orders(price)

            # Update tracking for next step
            for agent in self.agents:
                agent.update_last_price(price)
                agent.update_unrealized_pnl(price)

        return self.exchange.trade_log
