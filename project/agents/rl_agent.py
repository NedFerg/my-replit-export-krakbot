import collections
import random

import math

import torch
import torch.nn as nn
import torch.optim as optim

from agents.trader_agent import TraderAgent


# ---------------------------------------------------------------------------
# Noisy linear layer (NoisyNet)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """
    Linear layer with parametric noise on weights and biases.

    During training each forward pass adds Gaussian noise scaled by learned
    sigma parameters, giving the network state-dependent stochasticity.
    During evaluation (inference / target network) only the mean parameters
    (weight_mu, bias_mu) are used, making the output deterministic.

    Noise buffers are refreshed by calling reset_noise() — this happens once
    per training step in update_q() so that exploration varies every update.

    Parameters
    ----------
    sigma_init : initial value for all sigma parameters.  0.017 keeps the
                 initial noise small relative to a typical weight magnitude
                 of ±1/√fan_in, so early training is stable.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable mean and sigma parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise samples (not learnable — refreshed each step)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init)

    def reset_noise(self):
        """Sample fresh factorised Gaussian noise for weights and biases."""
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return torch.nn.functional.linear(x, w, b)


# ---------------------------------------------------------------------------
# Distributional Dueling Q-Network (C51 + Dueling + NoisyNets)
# ---------------------------------------------------------------------------

class DuelingQNetwork(nn.Module):
    """
    Distributional Dueling DQN.

    Instead of outputting a scalar Q(s,a) per action, the network outputs a
    probability distribution over num_atoms return atoms for every action:

        output shape: [batch, num_actions, num_atoms]   (raw logits)

    The dueling decomposition is applied atom-wise:
        Q_logits(s,a,z) = V(s,z) + A(s,a,z) − mean_a A(s,a,z)

    After softmax across atoms the expected Q-value is:
        Q(s,a) = Σ_z  softmax(Q_logits)[a,z] · support[z]

    Architecture
    ------------
    Shared:    Linear(input_dim, 64) → ReLU → Linear(64, 64) → ReLU
    Value:     NoisyLinear(64, 32) → ReLU → NoisyLinear(32, num_atoms)
    Advantage: NoisyLinear(64, 32) → ReLU → NoisyLinear(32, num_actions * num_atoms)
    """

    def __init__(self, input_dim, num_actions, num_atoms):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # Shared feature extractor (standard Linear — noise only in heads)
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Value stream — one atom-vector per state: [batch, num_atoms]
        self.value = nn.Sequential(
            NoisyLinear(64, 32),
            nn.ReLU(),
            NoisyLinear(32, num_atoms),
        )

        # Advantage stream — one atom-vector per (state, action): [batch, num_actions * num_atoms]
        self.advantage = nn.Sequential(
            NoisyLinear(64, 32),
            nn.ReLU(),
            NoisyLinear(32, num_actions * num_atoms),
        )

    def forward(self, x):
        f = self.feature(x)

        V = self.value(f)                                            # [batch, num_atoms]
        V = V.view(-1, 1, self.num_atoms)                           # [batch, 1, num_atoms]

        A = self.advantage(f)                                        # [batch, num_actions * num_atoms]
        A = A.view(-1, self.num_actions, self.num_atoms)            # [batch, num_actions, num_atoms]

        A_mean = A.mean(dim=1, keepdim=True)                        # [batch, 1, num_atoms]
        return V + (A - A_mean)                                      # [batch, num_actions, num_atoms]

    def reset_noise(self):
        """Resample noise buffers in every NoisyLinear layer."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ---------------------------------------------------------------------------
# Reinforcement-learning trader
# ---------------------------------------------------------------------------

class ReinforcementLearningTrader(TraderAgent):
    actions = ["buy", "sell", "hold"]

    # Reward shaping coefficients — class-level for easy tuning
    INVENTORY_PENALTY_COEF = 0.1  # penalty per unit of absolute position

    # Number of features produced by featurize_state().
    # Must match the length of the tuple featurize_state() returns.
    FEATURE_DIM = 13   # 8 base + 2 rolling vol (short_vol, long_vol) + 3 trend (mom_5, mom_20, mom_50)

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
        # C51 distributional RL hyper-parameters
        # The support is 51 evenly-spaced return atoms in [v_min, v_max].
        # v_min / v_max should bracket the realistic range of shaped rewards;
        # these can be tuned once observed return scales are known.
        # ---------------------------------------------------------------
        self.num_atoms = 51
        self.v_min = -20.0
        self.v_max = 20.0
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        self.device = torch.device("cpu")
        self.support = self.support.to(self.device)

        self.q_net = DuelingQNetwork(
            self.feature_dim, len(self.actions), self.num_atoms
        ).to(self.device)
        self.target_q_net = DuelingQNetwork(
            self.feature_dim, len(self.actions), self.num_atoms
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        # Polyak rate for soft target-network updates (applied each replay step)
        self.target_update_tau = 0.01

        # Cumulative replay step counter (useful for diagnostics / future extensions)
        self.replay_steps = 0

        # How many times replay() has been called; used to throttle updates
        # to once every replay_freq calls (standard DQN practice).
        self._replay_call_count = 0
        self.replay_freq = 4          # run one gradient batch every 4 env steps

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

        # N-step return accumulation
        # n_step_buffer collects raw 1-step transitions; once N are gathered
        # (or the episode ends) they are collapsed into a single N-step
        # transition (s_0, a_0, R_n, s_n) and pushed to replay_buffer.
        self.n_step = 3
        self.n_step_buffer = collections.deque(maxlen=self.n_step)

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
        self.n_step_buffer.clear()   # prevent transitions leaking across episodes
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
        B. vol_bucket    – current regime volatility parameter, 3dp.
        C. drift_bucket  – current drift, 3dp.
        D. inv_bucket    – agent's integer position (raw).
        E. regime        – regime string (converted to float in state_to_vector).
        F. m1            – 1-step momentum if MarketState provides it, else 0.
        G. m3            – 3-step momentum if MarketState provides it, else 0.
        H. imbalance     – order-flow imbalance if MarketState provides it, else 0.
        I. short_vol     – 5-step realized volatility from return history, capped at 0.1.
        J. long_vol      – 20-step realized volatility from return history, capped at 0.1.
        K. mom_5         – 5-bar price momentum, clipped to ±0.2.
        L. mom_20        – 20-bar price momentum, clipped to ±0.2.
        M. mom_50        – 50-bar price momentum, clipped to ±0.2.
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

        # Rolling realized volatility injected by the simulation each step.
        # Capped at 0.1 to keep the feature in a consistent magnitude range
        # regardless of extreme price moves.
        short_vol = round(min(getattr(market_state, "short_vol", 0.0), 0.1), 5)
        long_vol  = round(min(getattr(market_state, "long_vol",  0.0), 0.1), 5)

        # Multi-timeframe trend momentum injected by the simulation each step.
        # Clipped to ±0.2 so extreme short-term moves don't dominate the input.
        mom_5  = max(min(getattr(market_state, "mom_5",  0.0), 0.2), -0.2)
        mom_20 = max(min(getattr(market_state, "mom_20", 0.0), 0.2), -0.2)
        mom_50 = max(min(getattr(market_state, "mom_50", 0.0), 0.2), -0.2)

        return (
            price_bucket, vol_bucket, drift_bucket, inv_bucket, regime,
            m1, m3, imbalance,
            short_vol, long_vol,
            mom_5, mom_20, mom_50,
        )

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

    def _dist_to_q_values(self, logits):
        """
        Convert raw distribution logits to expected Q-values.

        logits : [batch, num_actions, num_atoms]  (raw network output)
        Returns : [batch, num_actions]  (expected Q per action)

        Softmax across atoms gives probabilities; dot-product with the
        support gives the expectation E[Z] = Σ_z p(z) · z.
        """
        probs = torch.softmax(logits, dim=-1)           # [batch, num_actions, num_atoms]
        support = self.support.view(1, 1, -1)           # [1, 1, num_atoms]
        return (probs * support).sum(dim=-1)            # [batch, num_actions]

    def q_value(self, state_vec, action):
        """
        Expected Q(s, a) from the live network (no grad, inference only).

        Returns a plain Python float.
        """
        idx = self.action_to_index[action]
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.q_net(x)                      # [1, num_actions, num_atoms]
            q_vals = self._dist_to_q_values(logits)     # [1, num_actions]
        return q_vals[0, idx].item()

    def target_q_value(self, state_vec, action):
        """
        Expected Q(s, a) from the target network (no grad, inference only).

        Used exclusively for action evaluation in update_q() so that the
        bootstrap signal comes from a stable, slowly-updated copy.
        """
        idx = self.action_to_index[action]
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.target_q_net(x)
            q_vals = self._dist_to_q_values(logits)
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

    def add_experience(self, prev_state, action, reward, new_state, done=False):
        """
        Accumulate a raw 1-step transition into the N-step buffer, then
        collapse and push to the replay buffer once N steps are available
        (or the episode ends).

        N-step return
        -------------
        R_n = Σ_{k=0}^{N-1} γ^k · r_{t+k}   (sum stops early if done=True)
        s_n = state reached after N steps (or at episode end)

        The collapsed transition (s_0, a_0, R_n, s_n) is stored in
        replay_buffer exactly like a 1-step transition, so replay() and
        update_q() require no changes — they simply receive a richer reward
        signal and a further-ahead bootstrap state.

        Priority
        --------
        New entries receive the current maximum priority so they are sampled
        at least once before being re-scored by actual TD error.
        """
        self.n_step_buffer.append((prev_state, action, reward, new_state, done))

        # Wait until the buffer is full (or the episode is done)
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Compute discounted N-step return, stopping at any terminal step
        R = 0.0
        gamma_pow = 1.0
        for (_, _, r, _, d) in self.n_step_buffer:
            R += gamma_pow * r
            if d:
                break
            gamma_pow *= self.gamma

        s0, a0, _, _, _ = self.n_step_buffer[0]
        s_n = self.n_step_buffer[-1][3]  # state after the last collected step

        transition = (s0, a0, R, s_n)
        max_prio = max((p for (_, p) in self.replay_buffer), default=1.0)
        self.replay_buffer.append((transition, max_prio))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

        if done:
            self.n_step_buffer.clear()

    def replay(self):
        """
        Sample a priority-weighted mini-batch and train the Q-network.

        Sampling is proportional to priority^alpha (Prioritized Experience
        Replay).  For each sampled transition:
          1. update_eligibilities() — bookkeeping; traces decay as usual.
          2. update_q()             — one backprop step; returns td_error.
          3. Refresh buffer priority to |td_error| + ε.

        No-ops until batch_size transitions are available or on non-update steps.
        """
        self._replay_call_count += 1
        if self._replay_call_count % self.replay_freq != 0:
            return
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
        C51 distributional update with Double-Q, NoisyNets, N-step returns,
        and Polyak target-network update.

        Algorithm
        ---------
        1. Build state tensors.

        2. Live network forward (with gradient):
               logits     → [1, num_actions, num_atoms]
               log_probs  → log_softmax over atoms
               log_probs_a → log-probs for the taken action: [1, num_atoms]

        3. Double-Q next-action selection (no gradient):
               live net  → expected Q per action → argmax → best_next_idx
               target net → probability distribution for best_next_idx

        4. C51 distributional projection (N-step Bellman):
               Tz_j = clamp(r + γ^n · z_j, v_min, v_max)
               b_j  = (Tz_j − v_min) / δz
               Spread p(z_j) proportionally to floor(b_j) and ceil(b_j)

        5. Cross-entropy loss: −Σ_j target_dist_j · log_probs_a_j
           Backprop → Adam step.

        6. Resample NoisyNet noise, then soft-update target network (Polyak).

        7. Return |ΔQ| as the PER priority signal.
        """
        s_vec = self.state_to_vector(prev_state)
        s2_vec = self.state_to_vector(new_state)

        x  = torch.tensor(s_vec,  dtype=torch.float32, device=self.device).unsqueeze(0)
        x2 = torch.tensor(s2_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_idx = torch.tensor([self.action_to_index[action]], device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # --- Live network forward (gradient tracked) ---
        logits = self.q_net(x)                                     # [1, num_actions, num_atoms]
        log_probs   = torch.log_softmax(logits, dim=-1)            # [1, num_actions, num_atoms]
        log_probs_a = log_probs[0, a_idx[0]]                       # [num_atoms]

        # --- Double-Q: live selects action, target evaluates distribution ---
        with torch.no_grad():
            next_logits_live  = self.q_net(x2)                     # [1, num_actions, num_atoms]
            next_q_vals_live  = self._dist_to_q_values(next_logits_live)   # [1, num_actions]
            best_next_idx     = next_q_vals_live.argmax(dim=1)     # [1]

            next_logits_target = self.target_q_net(x2)             # [1, num_actions, num_atoms]
            next_probs_target  = torch.softmax(next_logits_target, dim=-1)
            next_probs_best    = next_probs_target[0, best_next_idx[0]]    # [num_atoms]

        # --- C51 distributional projection (N-step Bellman operator) ---
        # Vectorised: no Python loop over atoms.
        gamma_n = self.gamma ** self.n_step
        Tz = r.unsqueeze(1) + gamma_n * self.support.view(1, -1)  # [1, num_atoms]
        Tz = Tz.clamp(self.v_min, self.v_max)

        b  = (Tz - self.v_min) / self.delta_z                     # [1, num_atoms]
        l  = b.floor().long().view(-1)                             # [num_atoms]
        u  = b.ceil().long().view(-1)                              # [num_atoms]
        b  = b.view(-1)                                            # [num_atoms]

        # Split next_probs_best mass between the two surrounding atoms.
        # index_add_ accumulates into duplicate indices correctly.
        target_dist = torch.zeros(self.num_atoms, device=self.device)
        target_dist.index_add_(0, l, next_probs_best * (u.float() - b))
        target_dist.index_add_(0, u, next_probs_best * (b - l.float()))

        # --- Cross-entropy loss and backprop ---
        loss = -(target_dist * log_probs_a).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Resample noise and soft-update target network ---
        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        with torch.no_grad():
            for param, target_param in zip(
                self.q_net.parameters(), self.target_q_net.parameters()
            ):
                target_param.data.mul_(1.0 - self.target_update_tau)
                target_param.data.add_(self.target_update_tau * param.data)

        # Cross-entropy loss is the natural C51 priority signal:
        # high loss ↔ large surprise ↔ should be revisited (replaces |ΔQ|).
        return loss.item()  # used by replay() to refresh PER priorities
