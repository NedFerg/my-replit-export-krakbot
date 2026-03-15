import collections
import math
import random

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

        # --- Rolling volatility tracking ----------------------------------
        # 1-step log-returns are stored; vols are computed from these each step.
        self.short_vol_window = 5
        self.long_vol_window = 20
        self.return_history = collections.deque(maxlen=self.long_vol_window)
        self._prev_price = None   # tracks last price for return computation

        # --- Risk-aware reward: RL-agent equity tracking -----------------
        # equity_peak and equity_history are reset each episode because a
        # fresh Simulation is constructed per episode in main.py.
        self.equity_peak = None
        self.equity_history = []

        # --- Multi-timeframe trend windows --------------------------------
        self.trend_windows = [5, 20, 50]
        # price_history (defined in __init__ above) stores raw prices per step
        # and is reused here for momentum lookback.

        # --- Volume and order-flow tracking ------------------------------
        self.volume_history = []
        self.buy_volume_history = []
        self.sell_volume_history = []
        self.volume_window = 20

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

            # --- Rolling realized volatility -----------------------------
            # Compute 1-step return and update the rolling return history.
            # Then attach short_vol and long_vol as extra state attributes
            # so agents can consume them without changing the broker or
            # MarketState class.
            if self._prev_price is not None and self._prev_price > 0:
                price_return = (price - self._prev_price) / self._prev_price
            else:
                price_return = 0.0
            self.return_history.append(price_return)
            self._prev_price = price

            if len(self.return_history) >= self.short_vol_window:
                short_slice = list(self.return_history)[-self.short_vol_window:]
                short_mean = sum(short_slice) / len(short_slice)
                short_var = sum((r - short_mean) ** 2 for r in short_slice) / max(len(short_slice) - 1, 1)
                state.short_vol = math.sqrt(short_var)
            else:
                state.short_vol = 0.0

            if len(self.return_history) >= self.long_vol_window:
                long_slice = list(self.return_history)
                long_mean = sum(long_slice) / len(long_slice)
                long_var = sum((r - long_mean) ** 2 for r in long_slice) / max(len(long_slice) - 1, 1)
                state.long_vol = math.sqrt(long_var)
            else:
                state.long_vol = 0.0

            # --- Multi-timeframe momentum ---------------------------------
            # price_history does NOT yet contain the current price (it is
            # appended at the end of the loop), so self.price_history[-w]
            # is exactly the price w steps ago — no off-by-one adjustment.
            for w in self.trend_windows:
                if len(self.price_history) >= w:
                    past_price = self.price_history[-w]
                    mom = (price - past_price) / past_price if past_price != 0 else 0.0
                else:
                    mom = 0.0
                setattr(state, f"mom_{w}", mom)

            # --- Volume and order-flow simulation -------------------------
            # prev_price: use the last entry in price_history (current price
            # is not appended until end of loop, so [-1] is one step back).
            _prev_p = self.price_history[-1] if self.price_history else price
            _price_move = abs(price - _prev_p)

            # Base volume proportional to absolute price move
            # (more volatility → more microstructure activity).
            base_vol = max(1.0, _price_move * 1000)
            volume   = base_vol * random.uniform(0.8, 1.2)

            # Split directionally: whichever side drove the move gets
            # a majority share (55–75%); the other side absorbs the rest.
            if price > _prev_p:
                buy_vol  = volume * random.uniform(0.55, 0.75)
                sell_vol = volume - buy_vol
            elif price < _prev_p:
                sell_vol = volume * random.uniform(0.55, 0.75)
                buy_vol  = volume - sell_vol
            else:
                buy_vol  = volume * 0.5
                sell_vol = volume * 0.5

            self.volume_history.append(volume)
            self.buy_volume_history.append(buy_vol)
            self.sell_volume_history.append(sell_vol)

            # Rolling average volume over the window
            if len(self.volume_history) >= self.volume_window:
                vol_slice   = self.volume_history[-self.volume_window:]
                rolling_vol = sum(vol_slice) / len(vol_slice)
            else:
                rolling_vol = 0.0

            # Imbalance: signed difference between buy and sell pressure
            vol_imbalance = buy_vol - sell_vol

            # Pressure: imbalance normalised to [−1, +1]
            _denom   = buy_vol + sell_vol
            pressure = vol_imbalance / _denom if _denom > 0 else 0.0

            state.rolling_vol   = rolling_vol
            state.vol_imbalance = vol_imbalance
            state.pressure      = pressure

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

                    # --- Equity peak / drawdown ---------------------------
                    self.equity_history.append(new_equity)
                    if self.equity_peak is None:
                        self.equity_peak = new_equity
                    if new_equity > self.equity_peak:
                        self.equity_peak = new_equity
                    state.drawdown = max(0.0, self.equity_peak - new_equity)

                    # --- Rolling equity-return volatility -----------------
                    # Computed over the last vol_window 1-step equity returns.
                    # Returns < 2 data points → vol is zero (no variance yet).
                    _vol_window = 20
                    if len(self.equity_history) >= 2:
                        _returns = []
                        for _i in range(1, len(self.equity_history)):
                            _prev = self.equity_history[_i - 1]
                            _curr = self.equity_history[_i]
                            if _prev != 0:
                                _returns.append((_curr - _prev) / _prev)
                        if len(_returns) >= _vol_window:
                            _slice = _returns[-_vol_window:]
                            _mean = sum(_slice) / len(_slice)
                            _var  = sum((r - _mean) ** 2 for r in _slice) / max(len(_slice) - 1, 1)
                            state.equity_vol = math.sqrt(_var)
                        else:
                            state.equity_vol = 0.0
                    else:
                        state.equity_vol = 0.0

                    if agent.prev_state is not None:
                        # Risk-aware reward: raw PnL minus drawdown and
                        # equity-volatility penalties.
                        # Coefficients (0.1, 0.05) are tunable starting points.
                        pnl_reward = new_equity - agent.prev_equity
                        reward = (
                            pnl_reward
                            - 0.1 * state.drawdown
                            - 0.05 * state.equity_vol
                        )
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
