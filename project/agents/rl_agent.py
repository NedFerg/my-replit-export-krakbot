import random
from agents.trader_agent import TraderAgent


class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    # Reward shaping coefficients — class-level for easy tuning
    INVENTORY_PENALTY_COEF = 0.1  # penalty per unit of absolute position

    def __init__(self, name, balance, latency=2):
        super().__init__(name, balance, latency)

        # Q-table and learning hyper-parameters (persist across episodes)
        # q_table structure: {state_tuple: {action: q_value}}
        self.q_table = {}
        self.alpha = 0.1    # learning rate
        self.gamma = 0.95   # discount factor

        # Eligibility traces for Q(λ)
        # Stored as {(state_tuple, action): eligibility_value}
        self.lambda_ = 0.8          # trace decay — higher = longer credit assignment
        self.eligibilities = {}     # reset each episode to prevent trace leakage

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

        Eligibilities are cleared per episode: traces accumulated in one
        episode should not propagate credit into the next.
        """
        super().reset_for_new_episode()
        self.last_mid_price = None
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None
        self.eligibilities.clear()
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
    # Eligibility traces
    # ------------------------------------------------------------------

    def update_eligibilities(self, state, action):
        """
        Maintain accumulating eligibility traces for Q(λ).

        Step 1 — decay all existing traces by γλ; prune those that have
        fallen below a negligible threshold to keep the dict compact.

        Step 2 — increment the trace for the (state, action) pair that
        was just visited (accumulating-trace variant).

        This means recently visited pairs receive a stronger update when
        the TD error is back-propagated in update_q(), giving the agent
        multi-step credit assignment without storing full trajectories.
        """
        decay = self.gamma * self.lambda_
        for key in list(self.eligibilities.keys()):
            self.eligibilities[key] *= decay
            if self.eligibilities[key] < 1e-6:
                del self.eligibilities[key]

        key = (state, action)
        self.eligibilities[key] = self.eligibilities.get(key, 0.0) + 1.0

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
        Sample a random mini-batch and apply Q(λ) updates.

        For each sampled transition:
          1. update_eligibilities() — decay existing traces and boost the
             visited (state, action) pair.
          2. update_q() — compute one TD error and propagate it through
             all currently eligible (state, action) pairs.

        No-ops until at least batch_size transitions have been collected,
        which avoids biased updates on a nearly-empty buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for prev_state, action, reward, new_state in batch:
            self.update_eligibilities(prev_state, action)
            self.update_q(prev_state, action, reward, new_state)

    # ------------------------------------------------------------------
    # Q(λ) core update
    # ------------------------------------------------------------------

    def update_q(self, prev_state, action, reward, new_state):
        """
        Q(λ) update: compute one TD error and broadcast it across all
        currently eligible (state, action) pairs weighted by their trace.

        TD error:
            δ = (reward + γ · max_a Q(new_state, a)) − Q(prev_state, action)

        Update rule for every eligible pair (s, a):
            Q(s, a) += α · δ · e(s, a)

        The q_table is a nested dict {state: {action: value}} so that
        decide() and encode_state() require no changes.  Entries for
        states not yet in the table are initialised to zero on first write.
        """
        # Compute the TD error from this specific transition
        current_q = self.q_table.get(prev_state, {}).get(action, 0.0)
        next_qvals = self.q_table.get(new_state, {})
        best_next = max(next_qvals.values()) if next_qvals else 0.0
        td_error = (reward + self.gamma * best_next) - current_q

        # Propagate TD error to all eligible (state, action) pairs
        for (s, a), e in self.eligibilities.items():
            if s not in self.q_table:
                self.q_table[s] = {act: 0.0 for act in self.actions}
            self.q_table[s][a] += self.alpha * td_error * e
