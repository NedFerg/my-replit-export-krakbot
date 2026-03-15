import random
from agents.trader_agent import TraderAgent


class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    def __init__(self, name, balance):
        super().__init__(name, balance)
        self.q_table = {}   # (price_bucket, spread_bucket, regime) → {action: Q-value}
        self.alpha = 0.1    # learning rate
        self.gamma = 0.95   # discount factor
        self.epsilon = 0.1  # exploration rate

        # Set by the simulation loop each step
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None

    def encode_state(self, market_state):
        """Discretize continuous market features into a hashable state tuple."""
        price_bucket = int(market_state.mid_price // 5)
        spread_bucket = int((market_state.spread or 0) // 1)
        regime = market_state.regime
        return (price_bucket, spread_bucket, regime)

    def decide(self, market_state):
        """Epsilon-greedy action selection over the current encoded state."""
        state = self.encode_state(market_state)

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
