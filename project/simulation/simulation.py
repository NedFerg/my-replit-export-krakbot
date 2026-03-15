from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.rl_agent import ReinforcementLearningTrader
from agents.market_agent import MarketAgent
from exchange.exchange import Exchange
from market_data.data_source import SimulatedDataSource
from risk.risk_manager import RiskManager, OrderIntent
from config.config import (
    SIMULATION_STEPS,
    INITIAL_BALANCE,
    MARKET_START_PRICE
)


class Simulation:
    """
    Orchestrates one episode of the multi-agent market simulation.

    Price generation is fully delegated to a MarketDataSource, so the
    simulation loop only consumes Tick objects — it has no knowledge of
    how prices are produced. Swapping SimulatedDataSource for a live or
    paper-trading source requires no changes here.
    """

    def __init__(self, agents=None, market_data_source=None, risk_manager=None):
        # --- Market data source ------------------------------------------
        # Create a default SimulatedDataSource if none is provided.
        if market_data_source is not None:
            self.market_data_source = market_data_source
        else:
            self.market_data_source = SimulatedDataSource(
                MarketAgent(MARKET_START_PRICE)
            )

        # Record the initial regime before any ticks are produced
        self.initial_regime = self.market_data_source.initial_regime

        # --- Order book --------------------------------------------------
        # Exchange no longer drives price; it only manages the order book.
        self.exchange = Exchange()

        # --- Agents -------------------------------------------------------
        if agents is not None:
            self.agents = agents
        else:
            self.agents = [
                ValueTrader("ValueTrader", INITIAL_BALANCE),
                MomentumTrader("MomentumTrader", INITIAL_BALANCE),
                RandomTrader("RandomTrader", INITIAL_BALANCE),
                ReinforcementLearningTrader("RLTrader", INITIAL_BALANCE),
            ]

        # --- Risk manager ------------------------------------------------
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()

        # --- Histories ---------------------------------------------------
        self.price_history = []
        self.regime_history = []

    def run(self):
        # Register agents with the risk manager: records starting equity and
        # clears per-episode counters (drawdown locks, rejection log, etc.)
        self.risk_manager.register_agents(self.agents)

        for _step in range(SIMULATION_STEPS):
            # --- Consume one tick from the data source -------------------
            tick = self.market_data_source.get_next_tick()
            price = tick.price

            # --- Build MarketState from the live order book + tick data --
            state = self.exchange.get_market_state(
                price,
                tick.regime,
                tick.drift,
                tick.volatility,
            )

            # --- Agent decisions + risk-gated order submission -----------
            step_actions = {}
            for agent in self.agents:
                action = agent.decide(state)
                step_actions[agent] = action  # record intent regardless of approval

                if action != "hold":
                    intent = OrderIntent(side=action, quantity=1, price=price)
                    if self.risk_manager.approve_order(agent, intent, state):
                        self.exchange.process_order(agent, action, price)
                    # else: silently blocked by risk manager

            # --- Fill any unmatched resting orders via market maker ------
            self.exchange.fill_resting_orders(price)

            # --- Tracking + Q-learning update ----------------------------
            for agent in self.agents:
                agent.update_last_price(state.mid_price)
                agent.update_unrealized_pnl(price)
                agent.record_equity()

                if isinstance(agent, ReinforcementLearningTrader):
                    new_encoded = agent.encode_state(state)
                    new_equity = agent.balance + agent.unrealized_pnl

                    if agent.prev_state is not None:
                        reward = new_equity - agent.prev_equity
                        agent.update_q(
                            agent.prev_state,
                            agent.prev_action,
                            reward,
                            new_encoded,
                        )

                    agent.prev_state = new_encoded
                    agent.prev_action = step_actions[agent]
                    agent.prev_equity = new_equity

            # --- Global risk check ---------------------------------------
            self.risk_manager.check_global_risk(self.agents)

            # --- Histories -----------------------------------------------
            self.price_history.append(price)
            self.regime_history.append(tick.regime)

        return self.exchange.trade_log, self.agents, self.price_history, self.regime_history
