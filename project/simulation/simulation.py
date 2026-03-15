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

            # Each agent decides and acts
            for agent in self.agents:
                agent.update_last_price(price)
                action = agent.decide(price)
                self.exchange.process_order(agent, action)

            for agent in self.agents:
                agent.update_unrealized_pnl(price)

        return self.exchange.trade_log
