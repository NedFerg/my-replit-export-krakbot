import random
from agents.trader_agent import TraderAgent


class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    # Reward shaping coefficients — class-level for easy tuning
    INVENTORY_PENALTY_COEF = 0.1  # penalty per unit of absolute position

    # Number of features produced by featurize_state().
    # Must match the length of the tuple featurize_state() returns.
    FEATURE_DIM = 8

    def __init__(self, name, balance, latency=2):
        super().__init__(name, balance, latency)

        # ---------------------------------------------------------------
        # Linear function approximator: Q(s,a) = w_a · x(s)
        #
        # One weight vector per action, each of length FEATURE_DIM.
        # Weights persist across episodes — they are the "memory" that
        # replaces the old tabular Q-table.
        # ---------------------------------------------------------------
        self.feature_dim = self.FEATURE_DIM
        self.weights = {a: [0.0] * self.feature_dim for a in self.actions}

        # Target network — same shape as main weights, updated slowly.
        # TD targets are computed from target_weights so they are stable
        # across consecutive updates (core DQN idea).
        self.target_weights = {a: w[:] for a, w in self.weights.items()}
        self.target_update_tau = 0.01       # Polyak rate for target network
        self.target_update_interval = 1     # update every N replay steps
        self.replay_steps = 0               # cumulative replay step counter

        # Learning hyper-parameters
        self.alpha = 0.01   # learning rate (smaller than tabular — LFA is noisier)
        self.gamma = 0.95   # discount factor

        # Eligibility traces for Q(λ)
        # Stored as {(state_tuple, action): eligibility_value}
        self.lambda_ = 0.8          # trace decay — higher = longer credit assignment
        self.eligibilities = {}     # reset each episode to prevent trace leakage

        # Polyak averaging coefficient for soft Q-target updates
        self.tau = 0.1

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

        Persisted across episodes: weight vectors, replay buffer.
        Epsilon decays here rather than being reset.
        Eligibilities are cleared: traces from one episode must not
        propagate credit into the next episode's weight updates.
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
    # State encoding (legacy — retained for reference)
    # ------------------------------------------------------------------

    def encode_state(self, market_state):
        """
        Legacy tabular encoding (7-tuple of integer/string buckets).
        Superseded by featurize_state() for the LFA pipeline.
        Retained for reference and potential diagnostics.
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
            momentum_bucket = 1 if delta > 0 else (-1 if delta < 0 else 0)
        else:
            momentum_bucket = 0

        inventory_bucket = max(-2, min(2, self.position))

        return (
            price_bucket, spread_bucket, regime,
            imbalance_bucket, vol_bucket, momentum_bucket, inventory_bucket,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def featurize_state(self, market_state, agent):
        """
        Convert raw market_state into a normalized feature tuple (length = FEATURE_DIM).

        Features
        --------
        A. price_bucket  – normalized 1-step price return, 3dp.
        B. vol_bucket    – current volatility, 3dp.
        C. drift_bucket  – current drift, 3dp.
        D. inv_bucket    – agent's integer position (raw).
        E. regime        – regime string (converted to float in state_to_vector).
        F. m1            – 1-step momentum if MarketState provides it, else 0.
        G. m3            – 3-step momentum if MarketState provides it, else 0.
        H. imbalance     – order-flow imbalance if MarketState provides it, else 0.

        Optional fields (m1, m3, imbalance) are gated with hasattr so the
        method is forward-compatible: adding those attributes to MarketState
        will activate them automatically.
        """
        prev_price = agent.last_mid_price
        if prev_price is not None and prev_price > 0:
            price_change = (market_state.mid_price - prev_price) / max(prev_price, 1e-6)
        else:
            price_change = 0.0
        price_bucket = round(price_change, 3)

        vol_bucket = round(market_state.volatility, 3)
        drift_bucket = round(market_state.drift, 3)
        inv_bucket = int(agent.position)
        regime = market_state.regime

        m1 = round(market_state.momentum_1, 3) if hasattr(market_state, "momentum_1") else 0
        m3 = round(market_state.momentum_3, 3) if hasattr(market_state, "momentum_3") else 0
        imbalance = (
            round(market_state.order_imbalance, 3)
            if hasattr(market_state, "order_imbalance") else 0
        )

        return (price_bucket, vol_bucket, drift_bucket, inv_bucket, regime, m1, m3, imbalance)

    # ------------------------------------------------------------------
    # Feature vector conversion
    # ------------------------------------------------------------------

    def state_to_vector(self, state_tuple):
        """
        Convert a featurize_state() tuple into a numeric list for dot-product
        computation.

          - Numeric elements (int, float) → cast to float unchanged.
          - String elements (regime) → stable hash embedding in [0, 1).

        The hash embedding is deterministic within a Python process, which is
        sufficient for learning inside a single run.
        """
        return [
            float(x) if isinstance(x, (int, float))
            else hash(x) % 1000 / 1000.0
            for x in state_tuple
        ]

    # ------------------------------------------------------------------
    # Q-value computation (linear approximation)
    # ------------------------------------------------------------------

    def q_value(self, state_vec, action):
        """
        Compute Q(s, a) = w_a · x(s) via dot product (live network).

        state_vec must be the output of state_to_vector() — a plain list of
        floats of length self.feature_dim.
        """
        w = self.weights[action]
        return sum(w[i] * state_vec[i] for i in range(self.feature_dim))

    def target_q_value(self, state_vec, action):
        """
        Compute Q(s, a) using the target network weights.

        Used exclusively for building TD targets in update_q() so that the
        bootstrap value is not moving during the weight update — this is the
        stabilisation mechanism from DQN.
        """
        w = self.target_weights[action]
        return sum(w[i] * state_vec[i] for i in range(self.feature_dim))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def decide(self, market_state):
        """
        Epsilon-greedy action selection using the linear Q-function.

        Greedy action: argmax_a Q(s, a) = argmax_a (w_a · x(s)).
        """
        state_tuple = self.featurize_state(market_state, self)

        # Update price reference for next step's featurize_state call
        self.last_mid_price = market_state.mid_price

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        state_vec = self.state_to_vector(state_tuple)
        return max(self.actions, key=lambda a: self.q_value(state_vec, a))

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def compute_reward(self, raw_reward, market_state):
        """
        Shape the raw equity-change reward before storing in the buffer.

        1. Inventory penalty — discourages large directional exposure.
        2. Volatility scaling — reduces reward magnitude in chaotic regimes.
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
        Maintain accumulating eligibility traces (state tuples as keys).

        Decays all existing traces by γλ, prunes negligible ones,
        then increments the current (state, action) pair by 1.
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
        Store a (state_tuple, action, reward, next_state_tuple) transition.

        When the buffer is full the oldest entry is evicted (FIFO).
        The buffer persists across episodes for cross-episode learning.
        """
        self.replay_buffer.append((prev_state, action, reward, new_state))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

    def replay(self):
        """
        Sample a random mini-batch and apply linear Q(λ) weight updates.

        For each sampled transition:
          1. update_eligibilities() — decay traces, boost current (s, a).
          2. update_q()             — compute soft TD error and propagate
                                      through all eligible pairs via weight
                                      gradient descent.

        No-ops until batch_size transitions are available.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for prev_state, action, reward, new_state in batch:
            self.update_eligibilities(prev_state, action)
            self.update_q(prev_state, action, reward, new_state)
            self.replay_steps += 1

    # ------------------------------------------------------------------
    # Linear Q(λ) weight update
    # ------------------------------------------------------------------

    def update_q(self, prev_state, action, reward, new_state):
        """
        Linear function-approximation Q(λ) update with Polyak-averaged targets.

        1. Convert state tuples to numeric feature vectors.

        2. Compute current Q and best next Q using the weight vectors:
               current_q  = w_action · x(prev_state)
               max_next_q = max_a  w_a · x(new_state)

        3. Soft TD target (Polyak averaging):
               td_target  = reward + γ · max_next_q
               soft_target = (1 − τ) · current_q + τ · td_target
               td_error   = soft_target − current_q
                           ≡ τ · (td_target − current_q)

        4. Gradient update for every eligible (s, a) pair, weighted by
           the pair's eligibility trace e(s, a):
               w_a[i] += α · td_error · e(s,a) · x_i(s)

        This is the semi-gradient TD(λ) update for linear Q-functions.
        Using Polyak averaging on the target damps instability when many
        (s, a) pairs are updated simultaneously through the traces.
        """
        s_vec = self.state_to_vector(prev_state)
        s2_vec = self.state_to_vector(new_state)

        current_q = self.q_value(s_vec, action)

        # Double-Q: live network selects the action, target network evaluates it.
        # This decouples selection from evaluation, reducing overestimation bias.
        best_next_action = max(self.actions, key=lambda a: self.q_value(s2_vec, a))
        max_next_q = self.target_q_value(s2_vec, best_next_action)

        td_target = reward + self.gamma * max_next_q
        soft_target = (1.0 - self.tau) * current_q + self.tau * td_target
        td_error = soft_target - current_q  # ≡ τ · (td_target − current_q)

        # Propagate through all eligible (state, action) pairs
        for (s, a), e in self.eligibilities.items():
            s_vec_e = self.state_to_vector(s)
            w = self.weights[a]
            step = self.alpha * td_error * e
            for i in range(self.feature_dim):
                w[i] += step * s_vec_e[i]

        # Soft-update target network toward main network (Polyak averaging)
        tau = self.target_update_tau
        for a in self.actions:
            tw = self.target_weights[a]
            mw = self.weights[a]
            for i in range(self.feature_dim):
                tw[i] = (1.0 - tau) * tw[i] + tau * mw[i]
