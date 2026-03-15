import random

import torch
import torch.nn as nn
import torch.optim as optim

from agents.trader_agent import TraderAgent


# ---------------------------------------------------------------------------
# Q-Network: small two-hidden-layer MLP
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Feedforward network mapping state features → Q-values for each action.

    Architecture: Linear(input_dim, 64) → ReLU → Linear(64, 64) → ReLU
                  → Linear(64, output_dim)

    output_dim == number of actions; the caller indexes into the output to
    obtain Q(s, a) for a specific action.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Reinforcement-learning trader
# ---------------------------------------------------------------------------

class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    # Reward shaping coefficients — class-level for easy tuning
    INVENTORY_PENALTY_COEF = 0.1  # penalty per unit of absolute position

    # Number of features produced by featurize_state().
    # Must match the length of the tuple featurize_state() returns.
    FEATURE_DIM = 8

    def __init__(self, name, balance, latency=2):
        super().__init__(name, balance, latency)

        self.feature_dim = self.FEATURE_DIM

        # ---------------------------------------------------------------
        # Action ↔ index mappings (needed for tensor indexing)
        # ---------------------------------------------------------------
        self.action_to_index = {a: i for i, a in enumerate(self.actions)}
        self.index_to_action = {i: a for a, i in self.action_to_index.items()}

        # ---------------------------------------------------------------
        # Neural Q-networks (live and target)
        #
        # q_net     — trained every replay step via backprop
        # target_q_net — slowly tracks q_net via Polyak averaging;
        #                used exclusively for bootstrap targets to keep
        #                them stable across consecutive updates (DQN idea)
        # ---------------------------------------------------------------
        self.device = torch.device("cpu")

        self.q_net = QNetwork(self.feature_dim, len(self.actions)).to(self.device)
        self.target_q_net = QNetwork(self.feature_dim, len(self.actions)).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Polyak rate for soft target-network updates (applied each replay step)
        self.target_update_tau = 0.01

        # Cumulative replay step counter (useful for diagnostics / future extensions)
        self.replay_steps = 0

        # ---------------------------------------------------------------
        # Learning hyper-parameters
        # ---------------------------------------------------------------
        self.gamma = 0.95   # discount factor

        # Polyak averaging coefficient for soft Q-target blending inside update_q
        self.tau = 0.1

        # Eligibility traces for Q(λ) — kept for structural consistency;
        # with a neural network the update is per-transition (see update_q).
        self.lambda_ = 0.8
        self.eligibilities = {}     # reset each episode

        # Epsilon-greedy exploration with per-episode decay
        self.epsilon = 0.10
        self.epsilon_decay = 0.98
        self.min_epsilon = 0.05

        # ---------------------------------------------------------------
        # Experience replay buffer (persists across episodes)
        # Each entry: (transition, priority)
        # transition = (prev_state, action, reward, new_state)
        # ---------------------------------------------------------------
        self.replay_buffer = []
        self.replay_capacity = 5000
        self.batch_size = 32

        # Prioritized Experience Replay (PER) hyper-parameters
        self.prioritized_alpha = 0.6    # how strongly priority biases sampling
        self.prioritized_epsilon = 1e-5  # floor added to |td_error|

        self.last_mid_price = None  # for momentum feature

        # Set by the simulation loop each step
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None

    def reset_for_new_episode(self):
        """
        Reset per-episode fields.

        Persisted across episodes: network weights, replay buffer.
        Epsilon decays here rather than being reset.
        Eligibilities are cleared: traces must not propagate across episode
        boundaries.
        """
        super().reset_for_new_episode()
        self.last_mid_price = None
        self.prev_state = None
        self.prev_action = None
        self.prev_equity = None
        self.eligibilities.clear()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # State encoding (legacy — retained for reference / diagnostics)
    # ------------------------------------------------------------------

    def encode_state(self, market_state):
        """
        Legacy tabular encoding (7-tuple of integer/string buckets).
        Superseded by featurize_state() + state_to_vector() for the DQN pipeline.
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
        Convert a featurize_state() tuple into a numeric list.

          - Numeric elements (int, float) → cast to float unchanged.
          - String elements (regime) → stable hash embedding in [0, 1).
        """
        return [
            float(x) if isinstance(x, (int, float))
            else hash(x) % 1000 / 1000.0
            for x in state_tuple
        ]

    # ------------------------------------------------------------------
    # Q-value helpers — neural network inference
    # ------------------------------------------------------------------

    def q_value(self, state_vec, action):
        """
        Compute Q(s, a) using the live network (inference only, no grad).

        state_vec — output of state_to_vector(), list of floats.
        Returns a plain Python float.
        """
        idx = self.action_to_index[action]
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(x)
        return q_vals[0, idx].item()

    def target_q_value(self, state_vec, action):
        """
        Compute Q(s, a) using the target network (inference only, no grad).

        Used exclusively inside update_q() to build stable TD targets —
        the target network lags the live network via Polyak averaging so
        the bootstrap value is not chasing a moving target.
        """
        idx = self.action_to_index[action]
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.target_q_net(x)
        return q_vals[0, idx].item()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def decide(self, market_state):
        """
        Epsilon-greedy action selection using the live Q-network.

        Greedy action: argmax_a Q(s, a) via live network.
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
        Shape the raw equity-change reward.

        1. Inventory penalty — discourages large directional exposure.
        2. Volatility scaling — reduces reward magnitude in chaotic regimes.
        """
        inventory_penalty = self.INVENTORY_PENALTY_COEF * abs(self.position)
        reward = raw_reward - inventory_penalty
        vol_scale = 1.0 / (1.0 + market_state.volatility)
        return reward * vol_scale

    # ------------------------------------------------------------------
    # Eligibility traces (structural — not used in neural weight update)
    # ------------------------------------------------------------------

    def update_eligibilities(self, state, action):
        """
        Maintain accumulating eligibility traces (state tuples as keys).

        Decays all existing traces by γλ, prunes negligible ones,
        then increments the current (state, action) pair by 1.

        Note: traces are tracked for structural consistency and future use.
        The neural network update in update_q() currently updates only the
        current (s, a) via backprop (full TD(λ) with backprop is non-trivial).
        """
        decay = self.gamma * self.lambda_
        for key in list(self.eligibilities.keys()):
            self.eligibilities[key] *= decay
            if self.eligibilities[key] < 1e-6:
                del self.eligibilities[key]

        key = (state, action)
        self.eligibilities[key] = self.eligibilities.get(key, 0.0) + 1.0

    # ------------------------------------------------------------------
    # Experience replay (Prioritized)
    # ------------------------------------------------------------------

    def add_experience(self, prev_state, action, reward, new_state):
        """
        Store a transition with maximum current priority.

        New transitions start at max priority so they are sampled at
        least once before being re-scored by actual TD error.
        Each entry: (transition, priority)  where
        transition = (prev_state, action, reward, new_state).
        """
        transition = (prev_state, action, reward, new_state)
        max_prio = max((p for (_, p) in self.replay_buffer), default=1.0)
        self.replay_buffer.append((transition, max_prio))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

    def replay(self):
        """
        Sample a priority-weighted mini-batch and train the Q-network.

        Sampling is proportional to priority^alpha (Prioritized Experience
        Replay).  For each sampled transition:
          1. update_eligibilities() — bookkeeping; traces decay as usual.
          2. update_q()             — one backprop step; returns td_error.
          3. Refresh buffer priority to |td_error| + ε.

        No-ops until batch_size transitions are available.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Build priority-weighted sampling distribution
        priorities = [p for (_, p) in self.replay_buffer]
        probs = [p ** self.prioritized_alpha for p in priorities]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample indices with replacement, weighted by priority
        indices = random.choices(range(len(self.replay_buffer)),
                                 weights=probs,
                                 k=self.batch_size)

        for idx in indices:
            transition, _ = self.replay_buffer[idx]
            prev_state, action, reward, new_state = transition

            self.update_eligibilities(prev_state, action)
            td_error = self.update_q(prev_state, action, reward, new_state)

            # Refresh priority from latest TD-error magnitude
            new_prio = abs(td_error) + self.prioritized_epsilon
            self.replay_buffer[idx] = (transition, new_prio)

            self.replay_steps += 1

    # ------------------------------------------------------------------
    # DQN training step (Double-Q + soft Q-target + Polyak target update)
    # ------------------------------------------------------------------

    def update_q(self, prev_state, action, reward, new_state):
        """
        One gradient step on the live Q-network for a single transition.

        Algorithm
        ---------
        1. Build input tensors from state vectors.

        2. Current Q-value (live network, with gradient):
               current_q = q_net(x)[action]

        3. Double-Q bootstrap target (no gradient):
               best_next_action = argmax_a  q_net(x2)      # live selects
               max_next_q       = target_q_net(x2)[best_next_action]  # target evaluates

        4. Soft TD target (Polyak blend of current and Bellman target):
               td_target  = reward + γ · max_next_q
               soft_target = (1 − τ) · current_q.detach() + τ · td_target
               td_error   = (soft_target − current_q).detach()

        5. MSE loss on (current_q, soft_target) → backprop → Adam step.

        6. Soft-update target network parameters toward live network:
               θ_target ← (1 − τ_target) · θ_target + τ_target · θ_live

        Returns td_error (float) so replay() can update PER priorities.
        """
        s_vec = self.state_to_vector(prev_state)
        s2_vec = self.state_to_vector(new_state)

        x = torch.tensor(s_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        x2 = torch.tensor(s2_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_idx = torch.tensor([self.action_to_index[action]], device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # Current Q(s, a) — live network, gradient flows through this
        q_vals = self.q_net(x)
        current_q = q_vals.gather(1, a_idx.unsqueeze(1)).squeeze(1)

        # Double-Q: live network selects next action, target network evaluates it
        with torch.no_grad():
            next_q_live = self.q_net(x2)
            best_next_idx = next_q_live.argmax(dim=1, keepdim=True)
            next_q_target = self.target_q_net(x2)
            max_next_q = next_q_target.gather(1, best_next_idx).squeeze(1)

        # Soft TD target (Polyak blend)
        td_target = r + self.gamma * max_next_q
        soft_target = (1.0 - self.tau) * current_q.detach() + self.tau * td_target
        td_error = (soft_target - current_q).detach().item()

        # Backprop on MSE(current_q, soft_target)
        loss = self.loss_fn(current_q, soft_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update target network toward live network (Polyak)
        with torch.no_grad():
            for param, target_param in zip(
                self.q_net.parameters(), self.target_q_net.parameters()
            ):
                target_param.data.mul_(1.0 - self.target_update_tau)
                target_param.data.add_(self.target_update_tau * param.data)

        return td_error  # used by replay() to refresh PER priorities
