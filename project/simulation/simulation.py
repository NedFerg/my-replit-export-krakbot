from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.rl_agent import ReinforcementLearningTrader
from agents.market_agent import MarketAgent
from exchange.exchange import Exchange
from risk.risk_manager import RiskManager, OrderIntent
from config.config import (
    SIMULATION_STEPS,
    INITIAL_BALANCE,
    MARKET_START_PRICE
)


class Simulation:
    def __init__(self, agents=None, risk_manager=None):
        # Fresh market and exchange each episode
        self.market = MarketAgent(MARKET_START_PRICE)
        self.exchange = Exchange(self.market)
        self.initial_regime = self.market.regime
        self.price_history = []
        self.regime_history = []

        # Accept externally-managed agents (e.g. for multi-episode training)
        # or create a default set if none provided.
        if agents is not None:
            self.agents = agents
        else:
            self.agents = [
                ValueTrader("ValueTrader", INITIAL_BALANCE),
                MomentumTrader("MomentumTrader", INITIAL_BALANCE),
                RandomTrader("RandomTrader", INITIAL_BALANCE),
                ReinforcementLearningTrader("RLTrader", INITIAL_BALANCE),
            ]

        # Accept an externally-managed RiskManager or create a default one
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()

    def run(self):
        # Register agents with the risk manager so it can track starting equity
        # and reset per-episode counters (drawdown locks, rejection log, etc.)
        self.risk_manager.register_agents(self.agents)

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

            # All agents decide.  For non-hold actions the order is submitted
            # only if the risk manager approves it.
            step_actions = {}
            for agent in self.agents:
                action = agent.decide(state)
                step_actions[agent] = action  # record intent regardless of approval

                if action != "hold":
                    intent = OrderIntent(side=action, quantity=1, price=price)
                    if self.risk_manager.approve_order(agent, intent, state):
                        self.exchange.process_order(agent, action, price)
                    # else: order silently blocked by risk manager

            # Fill any unmatched resting orders via market maker
            self.exchange.fill_resting_orders(price)

            # Update tracking and run Q-learning update for the RL agent
            for agent in self.agents:
                agent.update_last_price(state.mid_price)
                agent.update_unrealized_pnl(price)
                agent.record_equity()

                if isinstance(agent, ReinforcementLearningTrader):
                    new_encoded = agent.encode_state(state)
                    new_equity = agent.balance + agent.unrealized_pnl

                    # Skip Q-update on the very first step (no previous state yet)
                    if agent.prev_state is not None:
                        reward = new_equity - agent.prev_equity
                        agent.update_q(
                            agent.prev_state,
                            agent.prev_action,
                            reward,
                            new_encoded
                        )

                    # Store experience for next step's update
                    agent.prev_state = new_encoded
                    agent.prev_action = step_actions[agent]
                    agent.prev_equity = new_equity

            # Check global risk after all equity is updated
            self.risk_manager.check_global_risk(self.agents)

            # Record price and regime history
            self.price_history.append(price)
            self.regime_history.append(self.market.regime)

        return self.exchange.trade_log, self.agents, self.price_history, self.regime_history
