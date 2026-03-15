from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.rl_agent import ReinforcementLearningTrader
from agents.market_agent import MarketAgent
from broker.broker import SimulatedBroker
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

    All market data flows in through a MarketDataSource (Tick objects).
    All order execution flows out through a Broker, which returns Fill objects.
    The simulation has no direct knowledge of Exchange internals.
    """

    def __init__(self, agents=None, market_data_source=None, broker=None, risk_manager=None):
        # --- Market data source ------------------------------------------
        if market_data_source is not None:
            self.market_data_source = market_data_source
        else:
            self.market_data_source = SimulatedDataSource(
                MarketAgent(MARKET_START_PRICE)
            )

        self.initial_regime = self.market_data_source.initial_regime

        # --- Broker (order execution) ------------------------------------
        self.broker = broker if broker is not None else SimulatedBroker()

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
        # Register agents: records starting equity, clears per-episode counters
        self.risk_manager.register_agents(self.agents)

        for _step in range(SIMULATION_STEPS):
            # --- Consume one tick from the data source -------------------
            tick = self.market_data_source.get_next_tick()
            price = tick.price

            # --- Build MarketState via the broker ------------------------
            state = self.broker.get_market_state(
                price,
                tick.regime,
                tick.drift,
                tick.volatility,
            )

            # --- Agent decisions + risk-gated order submission -----------
            # The broker returns Fill objects with actual execution prices
            # (slippage-adjusted).  Agent state (balance, position, PnL) is
            # updated inside the Exchange during matching.
            step_actions = {}
            for agent in self.agents:
                action = agent.decide(state)
                step_actions[agent] = action  # record intent regardless of approval

                if action != "hold":
                    intent = OrderIntent(side=action, quantity=1, price=price)
                    if self.risk_manager.approve_order(agent, intent, state):
                        fills = self.broker.submit_order(agent, intent, state)
                        # fills carry (agent_name, side, exec_price, qty) —
                        # state already updated by Exchange during matching;
                        # available here for logging or richer RL rewards.

            # --- End-of-step settlement via broker -----------------------
            self.broker.fill_resting_orders(price)

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

        return self.broker.trade_log, self.agents, self.price_history, self.regime_history
