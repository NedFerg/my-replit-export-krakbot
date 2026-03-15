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

    Latency model
    -------------
    When an agent decides on an action, the resulting OrderIntent is not sent
    to the broker immediately.  Instead it is placed on a latency_queue with a
    delivery step of  current_step + agent.latency.  At the start of each step
    the queue is drained: any intent whose delivery step has been reached is
    risk-checked and forwarded to the broker using the *current* market state.
    This means a delayed order may be rejected (or fill at a stale price) if the
    market has moved before the order arrives — realistic behaviour for slow or
    high-latency agents.

    Fee model
    ---------
    Fees are applied inside the Exchange during matching and are deducted
    directly from the agent's balance.  Because equity = balance + unrealized_pnl,
    fees naturally reduce equity and flow into the RL reward signal without any
    extra reward shaping.
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

        # --- Latency queue: (deliver_step, agent, order_intent) ----------
        self.latency_queue = []

        # --- Histories ---------------------------------------------------
        self.price_history = []
        self.regime_history = []

    def run(self):
        # Register agents: records starting equity, clears per-episode counters
        self.risk_manager.register_agents(self.agents)

        for current_step in range(SIMULATION_STEPS):
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

            # --- Deliver queued orders that are due ----------------------
            # Use the *current* state for the risk check so a delayed order
            # is evaluated against up-to-date market conditions.
            due = [
                (s, a, i) for (s, a, i) in self.latency_queue
                if s <= current_step
            ]
            self.latency_queue = [
                (s, a, i) for (s, a, i) in self.latency_queue
                if s > current_step
            ]
            for _, agent, intent in due:
                if self.risk_manager.approve_order(agent, intent, state):
                    self.broker.submit_order(agent, intent, state)

            # --- Agent decisions → latency queue -------------------------
            step_actions = {}
            for agent in self.agents:
                action = agent.decide(state)
                step_actions[agent] = action

                if action != "hold":
                    intent = OrderIntent(side=action, quantity=1, price=price)
                    deliver_step = current_step + agent.latency
                    self.latency_queue.append((deliver_step, agent, intent))

            # --- End-of-step settlement via broker -----------------------
            self.broker.fill_resting_orders(price)

            # --- Tracking + Q-learning update ----------------------------
            for agent in self.agents:
                agent.update_last_price(state.mid_price)
                agent.update_unrealized_pnl(price)
                agent.record_equity()

                if isinstance(agent, ReinforcementLearningTrader):
                    new_encoded = agent.featurize_state(state, agent)
                    new_equity = agent.balance + agent.unrealized_pnl

                    if agent.prev_state is not None:
                        raw_reward = new_equity - agent.prev_equity
                        # Shape reward: inventory penalty + volatility scaling
                        reward = agent.compute_reward(raw_reward, state)
                        # Store in replay buffer then learn from a random batch
                        done = (current_step == SIMULATION_STEPS - 1)
                        agent.add_experience(
                            agent.prev_state,
                            agent.prev_action,
                            reward,
                            new_encoded,
                            done,
                        )
                        agent.replay()

                    agent.prev_state = new_encoded
                    agent.prev_action = step_actions[agent]
                    agent.prev_equity = new_equity

            # --- Global risk check ---------------------------------------
            self.risk_manager.check_global_risk(self.agents)

            # --- Histories -----------------------------------------------
            self.price_history.append(price)
            self.regime_history.append(tick.regime)

        return self.broker.trade_log, self.agents, self.price_history, self.regime_history
