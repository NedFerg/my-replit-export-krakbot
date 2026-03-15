import random
from agents.trader_agent import TraderAgent


class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    def __init__(self, name, balance):
        super().__init__(name, balance)
        self.q_table = {}   # rich state tuple → {action: Q-value}
        self.alpha = 0.1    # learning rate
        self.gamma = 0.95   # discount factor
        self.epsilon = 0.1  # exploration rate

        self.last_mid_price = None  # for momentum feature

        # Set by the simulation loop each step
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None

    def reset_for_new_episode(self):
        """Reset per-episode fields; Q-table persists across episodes."""
        super().reset_for_new_episode()
        self.last_mid_price = None
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None

    def encode_state(self, market_state):
        """
        Discretize seven market features into a hashable state tuple.

        Features:
          price_bucket    – mid-price in bands of 5
          spread_bucket   – spread rounded down to nearest integer
          regime          – string from MarketAgent
          imbalance_bucket – sign of (bid_size - ask_size): -1 / 0 / +1
          vol_bucket      – volatility scaled by 10 and truncated
          momentum_bucket – sign of price change since last step: -1 / 0 / +1
          inventory_bucket – clamped integer position in [-2, 2]
        """
        price_bucket = int(market_state.mid_price // 5)
        spread_bucket = int((market_state.spread or 0) // 1)
        regime = market_state.regime

        imbalance = market_state.bid_size - market_state.ask_size
        if imbalance < 0:
            imbalance_bucket = -1
        elif imbalance > 0:
            imbalance_bucket = 1
        else:
            imbalance_bucket = 0

        vol_bucket = int(market_state.volatility * 10)

        if self.last_mid_price is not None:
            delta = market_state.mid_price - self.last_mid_price
            if delta < 0:
                momentum_bucket = -1
            elif delta > 0:
                momentum_bucket = 1
            else:
                momentum_bucket = 0
        else:
            momentum_bucket = 0

        inventory_bucket = max(-2, min(2, self.position))

        return (
            price_bucket,
            spread_bucket,
            regime,
            imbalance_bucket,
            vol_bucket,
            momentum_bucket,
            inventory_bucket,
        )

    def decide(self, market_state):
        """Epsilon-greedy action selection over the current encoded state."""
        state = self.encode_state(market_state)

        # Update momentum reference for next step's encode_state call
        self.last_mid_price = market_state.mid_price

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        qvals = self.q_table.get(state, {})
        if not qvals:
            return random.choice(self.actions)

        return max(qvals, key=qvals.get)

    def update_q(self, prev_state, action, reward, new_state):
        """Standard Q-learning (Bellman) update."""
        qvals = self.q_table.setdefault(prev_state, {a: 0.0 for a in self.actions})
        next_qvals = self.q_table.get(new_state, {a: 0.0 for a in self.actions})
        best_next = max(next_qvals.values())
        qvals[action] += self.alpha * (reward + self.gamma * best_next - qvals[action])
