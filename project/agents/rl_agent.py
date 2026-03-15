import random
from agents.trader_agent import TraderAgent


class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    # Reward shaping coefficients — class-level for easy tuning
    INVENTORY_PENALTY_COEF = 0.1  # penalty per unit of absolute position

    def __init__(self, name, balance, latency=2):
        super().__init__(name, balance, latency)

        # Q-table and learning hyper-parameters (persist across episodes)
        self.q_table = {}
        self.alpha = 0.1    # learning rate
        self.gamma = 0.95   # discount factor

        # Epsilon-greedy exploration with per-episode decay
        self.epsilon = 0.10
        self.epsilon_decay = 0.98
        self.min_epsilon = 0.05

        # Experience replay buffer (persists across episodes)
        self.replay_buffer = []
        self.replay_capacity = 5000
        self.batch_size = 32

        self.last_mid_price = None  # for momentum feature

        # Set by the simulation loop each step
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None

    def reset_for_new_episode(self):
        """
        Reset per-episode fields.

        Persisted across episodes: Q-table, replay buffer, epsilon
        (epsilon decays here rather than being reset).
        """
        super().reset_for_new_episode()
        self.last_mid_price = None
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None
        # Decay exploration rate — more exploitation as agent matures
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def encode_state(self, market_state):
        """
        Discretize seven market features into a hashable state tuple.

        Features:
          price_bucket     – mid-price in bands of 5
          spread_bucket    – spread rounded down to nearest integer
          regime           – string from MarketAgent
          imbalance_bucket – sign of (bid_size - ask_size): -1 / 0 / +1
          vol_bucket       – volatility scaled by 10 and truncated
          momentum_bucket  – sign of price change since last step: -1 / 0 / +1
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

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def compute_reward(self, raw_reward, market_state):
        """
        Shape the raw equity-change reward before storing in the buffer.

        1. Inventory penalty — discourages large directional exposure.
           The agent pays INVENTORY_PENALTY_COEF per unit of |position|
           regardless of PnL, nudging it toward flat positioning.

        2. Volatility scaling — divides the reward by (1 + volatility).
           In chaotic regimes the magnitude of accidental gains/losses is
           inflated; scaling down prevents the agent from over-fitting to
           lucky windfalls and encourages caution in noisy markets.
        """
        inventory_penalty = self.INVENTORY_PENALTY_COEF * abs(self.position)
        reward = raw_reward - inventory_penalty
        vol_scale = 1.0 / (1.0 + market_state.volatility)
        return reward * vol_scale

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def add_experience(self, prev_state, action, reward, new_state):
        """
        Store a transition in the replay buffer.

        When the buffer is full the oldest transition is evicted (FIFO).
        The buffer is intentionally not reset between episodes so the agent
        can learn from the full history of its interactions.
        """
        self.replay_buffer.append((prev_state, action, reward, new_state))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

    def replay(self):
        """
        Sample a random mini-batch from the buffer and apply Q-updates.

        No-ops until at least batch_size transitions have been collected,
        which avoids biased updates on a nearly-empty buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for prev_state, action, reward, new_state in batch:
            self.update_q(prev_state, action, reward, new_state)

    # ------------------------------------------------------------------
    # Q-learning core
    # ------------------------------------------------------------------

    def update_q(self, prev_state, action, reward, new_state):
        """Standard Q-learning (Bellman) update."""
        qvals = self.q_table.setdefault(prev_state, {a: 0.0 for a in self.actions})
        next_qvals = self.q_table.get(new_state, {a: 0.0 for a in self.actions})
        best_next = max(next_qvals.values())
        qvals[action] += self.alpha * (reward + self.gamma * best_next - qvals[action])
