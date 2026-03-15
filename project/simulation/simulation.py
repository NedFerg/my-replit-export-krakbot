from project.agents.trader_agent import TraderAgent
from project.agents.market_agent import MarketAgent
from project.exchange.exchange import Exchange
from project.config.config import (
    SIMULATION_STEPS,
    INITIAL_BALANCE,
    MARKET_START_PRICE
)

class Simulation:
    def __init__(self):
        self.market = MarketAgent(MARKET_START_PRICE)
        self.exchange = Exchange(self.market)
        self.agents = [
            TraderAgent("AgentA", INITIAL_BALANCE),
            TraderAgent("AgentB", INITIAL_BALANCE)
        ]

    def run(self):
        for step in range(SIMULATION_STEPS):
            # Update market price
            price = self.exchange.update_market()

            # Each agent decides and acts
            for agent in self.agents:
                action = agent.decide(price)
                self.exchange.process_order(agent, action)

        return self.exchange.trade_log
