import collections
import random
import math
import time
import json

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
    per training step in update_value() so that exploration varies every update.

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

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

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
# C51 Distributional Value Network  V(s)
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    C51 distributional state-value network.

    Outputs a probability distribution over num_atoms return atoms for the
    current state only — no action dimension.

        output shape: [batch, num_atoms]   (raw logits, softmax → probs)

    Architecture
    ------------
    Shared:  Linear(input_dim, 64) → ReLU → Linear(64, 64) → ReLU
    Value:   NoisyLinear(64, 32)   → ReLU → NoisyLinear(32, num_atoms)
    """

    def __init__(self, input_dim, num_atoms):
        super().__init__()
        self.num_atoms = num_atoms

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            NoisyLinear(64, 32),
            nn.ReLU(),
            NoisyLinear(32, num_atoms),
        )

    def forward(self, x):
        return self.value(self.feature(x))   # [batch, num_atoms]

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ---------------------------------------------------------------------------
# Gaussian Actor Network
# ---------------------------------------------------------------------------

class ActorNetwork(nn.Module):
    """
    Stochastic Gaussian actor for continuous multi-asset portfolio exposure.

    Input  : state feature vector (FEATURE_DIM,)
    Output : mu, log_std — each of shape (num_assets,)
    Action : tanh(sample from N(mu, std)) — each component in [-1, +1]

    The tanh squash ensures every target exposure is bounded without hard
    clipping, and the log-probability correction (SAC-style) accounts for the
    change of variables so gradient estimates are unbiased.
    """

    LOG_STD_MIN = -4.0   # clamp: prevents std → 0 (degenerate deterministic)
    LOG_STD_MAX =  2.0   # clamp: prevents std → ∞ (random walk)

    def __init__(self, input_dim, num_assets):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(64, num_assets)
        self.log_std_head = nn.Linear(64, num_assets)

    def forward(self, x):
        h = self.net(x)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, x):
        """
        Reparameterised sample + log-probability.

        Returns
        -------
        action   : [batch, num_assets] — tanh-squashed, in [-1, +1]
        log_prob : [batch]             — sum of per-asset log probs with
                                         tanh Jacobian correction
        """
        mu, log_std = self.forward(x)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        raw  = dist.rsample()                                    # reparameterised
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)                          # scalar per sample
        return action, log_prob


# ---------------------------------------------------------------------------
# Reinforcement-learning trader — multi-asset actor-critic
# ---------------------------------------------------------------------------

class ReinforcementLearningTrader(TraderAgent):
    IS_RL_AGENT = True   # tells the exchange not to mutate self.position

    # Assets the portfolio covers — 8 total.  BTC and ETH are anchors/hedges;
    # all 8 have continuous exposure targets in [-1, +1].
    PORTFOLIO_ASSETS = ["SOL", "XRP", "LINK", "ETH", "BTC", "XLM", "HBAR", "AVAX"]

    # Reward shaping coefficients — class-level for easy tuning
    INVENTORY_PENALTY_COEF = 0.1  # penalty per unit of mean abs portfolio exposure

    # Number of features produced by featurize_state().
    # Must match the length of the tuple featurize_state() returns.
    FEATURE_DIM = 88   # 16 SOL-only + 64 cross-asset (8×8) + 7 crypto-regime + 1 macro_regime

    def __init__(self, name, balance, latency=2, broker=None, dry_run=True):
        super().__init__(name, balance, latency)

        # Live-broker wiring — None in pure-simulation mode
        self.broker  = broker
        self.dry_run = dry_run
        self.ready   = False   # set True by warm_up() before live trading begins

        self.feature_dim = self.FEATURE_DIM

        # Multi-asset portfolio -------------------------------------------
        self.assets     = list(self.PORTFOLIO_ASSETS)
        self.num_assets = len(self.assets)

        # Per-asset current exposures (floats in [-1, +1]).
        # self.position (from TraderAgent) is kept in sync with SOL.
        self.positions        = {a: 0.0 for a in self.assets}
        self.target_exposures = {a: 0.0 for a in self.assets}

        # Legacy single-asset compatibility (SOL):
        self.target_exposure = 0.0   # kept in sync with target_exposures["SOL"]

        # Per-asset exposure caps — reflect the desired risk weighting:
        # SOL > XRP > LINK > ETH > BTC (SOL is the highest-conviction asset;
        # BTC is treated as a hedge so its max short is also the smallest).
        self.max_long = {
            "SOL":  1.0,
            "XRP":  0.8,
            "LINK": 0.7,
            "ETH":  0.6,
            "BTC":  0.3,
            "XLM":  0.6,
            "HBAR": 0.6,
            "AVAX": 0.7,
        }
        self.max_short = {
            "SOL":  -0.5,
            "XRP":  -0.4,
            "LINK": -0.4,
            "ETH":  -0.3,
            "BTC":  -0.2,
            "XLM":  -0.3,
            "HBAR": -0.3,
            "AVAX": -0.4,
        }

        # ---------------------------------------------------------------
        # C51 distributional RL hyper-parameters
        # ---------------------------------------------------------------
        self.num_atoms = 51
        self.v_min     = -20.0
        self.v_max     =  20.0
        self.delta_z   = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support   = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        self.device  = torch.device("cpu")
        self.support = self.support.to(self.device)

        # ---------------------------------------------------------------
        # Value network  V(s)  (C51 distributional)
        # ---------------------------------------------------------------
        self.value_net = ValueNetwork(self.feature_dim, self.num_atoms).to(self.device)
        self.target_value_net = ValueNetwork(self.feature_dim, self.num_atoms).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.eval()

        self.critic_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)

        # Polyak rate for soft target-network updates
        self.target_update_tau = 0.01

        # ---------------------------------------------------------------
        # Actor network  π(a|s)  (Gaussian + tanh squash)
        # ---------------------------------------------------------------
        ACTOR_LR = 3e-4
        self.actor = ActorNetwork(self.feature_dim, self.num_assets).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # Log-probability of the last sampled action; used for the online
        # actor update in the step immediately following the action.
        self.last_log_prob = None
        self.prev_log_prob = None

        # Cumulative replay step counter
        self.replay_steps = 0
        self._replay_call_count = 0
        self.replay_freq = 4   # gradient update every 4 env steps

        # ---------------------------------------------------------------
        # Learning hyper-parameters
        # ---------------------------------------------------------------
        self.gamma   = 0.95
        self.tau     = 0.1
        self.lambda_ = 0.8
        self.eligibilities = {}

        # Epsilon attribute kept for display / diagnostics.
        # Actor-based policy has no ε-greedy; shown as 0.
        self.epsilon       = 0.0
        self.epsilon_decay = 1.0
        self.min_epsilon   = 0.0

        # ---------------------------------------------------------------
        # Prioritized Experience Replay + N-step returns
        # ---------------------------------------------------------------
        self.replay_buffer   = []
        self.replay_capacity = 5000
        self.batch_size      = 32

        self.prioritized_alpha   = 0.6
        self.prioritized_epsilon = 1e-5

        self.n_step        = 3
        self.n_step_buffer = collections.deque(maxlen=self.n_step)

        self.last_mid_price = None
        self.prev_state     = None
        self.prev_action    = None
        self.prev_equity    = None

    # ------------------------------------------------------------------
    # Warm-up and live step
    # ------------------------------------------------------------------

    def warm_up(self, broker, cycles=5):
        """
        Warm-up phase before live trading begins.

        Syncs account state, emits heartbeats, and records health metrics
        for `cycles` iterations.  Sets self.ready = True on completion so
        that step() will begin executing trades.
        """
        print(f"[WARM-UP] Starting warm-up for {cycles} cycles")

        stable = 0
        while stable < cycles:
            # Abort warm-up immediately if kill switch has been triggered
            if getattr(broker, "kill_switch", False):
                print("[WARM-UP] Kill switch active — aborting warm-up")
                return

            state = broker.sync_live_account_state()
            if state is None:
                print("[WARM-UP] Account sync failed — retrying")
                time.sleep(1)
                continue

            balances, positions = state

            # balances is a dict — None means API failure, {} means zero balance (valid)
            if balances is None:
                print("[WARM-UP] Empty balances — retrying")
                time.sleep(1)
                continue

            broker.heartbeat()
            broker.record_health_metrics()

            stable += 1
            print(f"[WARM-UP] Cycle {stable}/{cycles} complete")
            time.sleep(1)

        print("[WARM-UP] Warm-up complete — agent is now active")
        self.ready = True

    # ------------------------------------------------------------------
    # Live state construction helpers
    # ------------------------------------------------------------------

    def _update_price_history(self, live_prices: dict):
        """Maintain a rolling 50-bar price history for each asset."""
        if not hasattr(self, "_price_history"):
            self._price_history = {a: [] for a in self.assets}
        for a in self.assets:
            p = live_prices.get(a, 0.0)
            if p > 0:
                self._price_history[a].append(p)
                if len(self._price_history[a]) > 50:
                    self._price_history[a].pop(0)

    def _momentum(self, asset: str, n: int) -> float:
        """n-bar price momentum, clipped to ±0.2."""
        hist = getattr(self, "_price_history", {}).get(asset, [])
        if len(hist) < n + 1:
            return 0.0
        ret = (hist[-1] - hist[-(n + 1)]) / max(hist[-(n + 1)], 1e-6)
        return max(min(ret, 0.2), -0.2)

    def _build_live_state_vector(self, live_prices: dict) -> list:
        """
        Build the 88-dimensional feature vector from live broker prices.
        Missing simulation features (vol, drift, order flow) are zeroed —
        the network was designed to handle sparse inputs gracefully.
        """
        sol_price = live_prices.get("SOL", 0.0)
        hist_sol  = getattr(self, "_price_history", {}).get("SOL", [])
        if len(hist_sol) >= 2 and hist_sol[-2] > 0:
            price_change = (sol_price - hist_sol[-2]) / hist_sol[-2]
        else:
            price_change = 0.0

        # SOL-specific features (16)
        features = [
            round(price_change, 3),   # price_bucket
            0.0,                      # vol_bucket (simulation-only)
            0.0,                      # drift_bucket (simulation-only)
            float(self.target_exposures.get("SOL", 0.0)),  # inv_bucket
            0.0,                      # regime (0 = neutral/unknown)
            0.0, 0.0, 0.0,           # m1, m3, imbalance
            0.0, 0.0,                 # short_vol, long_vol
            self._momentum("SOL",  5),
            self._momentum("SOL", 20),
            self._momentum("SOL", 50),
            0.0, 0.0, 0.0,           # rolling_vol, vol_imbalance, pressure
        ]

        # Cross-asset features (64 = 8 × 8)
        for a in self.assets:
            p = live_prices.get(a, 0.0)
            features.extend([
                p / 1000.0,               # normalized price
                0.0,                      # vol
                self._momentum(a,  5) / 100.0,
                self._momentum(a, 20) / 100.0,
                self._momentum(a, 50) / 100.0,
                0.0, 0.0, 0.0,           # rolling_vol, vol_imbalance, pressure
            ])

        # Crypto-regime features (7) + macro_regime (1)
        features.extend([0.0] * 8)

        return features

    def step(self):
        """
        Live-trading entry point called by run_live_trading_loop().

        Guards against trading before warm_up() completes.
        Refreshes live prices via the broker, runs the actor network to
        produce per-asset exposure targets, and logs the decisions with
        direction labels and confidence scores.
        """
        if not self.ready:
            print("[WARM-UP] Agent not ready — skipping step()")
            return

        # Refresh prices if a live broker is wired
        if self.broker is not None:
            self.broker.update_prices_if_needed()

        live_prices = getattr(self.broker, "live_prices", {}) if self.broker else {}
        if not live_prices:
            print("[STEP] No live prices available — skipping decision")
            return

        # Update rolling price history for momentum computation
        self._update_price_history(live_prices)

        # Build 88-dim state vector and run actor network
        state_vec = self._build_live_state_vector(live_prices)
        x = torch.tensor(state_vec, dtype=torch.float32,
                          device=self.device).unsqueeze(0)   # [1, feature_dim]

        self.actor.eval()
        with torch.no_grad():
            mu, log_std = self.actor.forward(x)              # [1, num_assets] each
            std = log_std.exp()

        mu_list  = mu.squeeze(0).tolist()
        std_list = std.squeeze(0).tolist()

        # Sample action for target exposures (with grad for training later)
        self.actor.train()
        action, log_prob = self.actor.sample(x)
        self.last_log_prob = log_prob

        action_list = action.detach().squeeze(0).tolist()
        for i, a in enumerate(self.assets):
            self.target_exposures[a] = float(action_list[i])
        self.target_exposure = self.target_exposures["SOL"]

        # ----------------------------------------------------------------
        # Per-asset decision log
        # ----------------------------------------------------------------
        HOLD_THRESH = 0.05   # |action| below this → HOLD
        header = (f"[STEP] prices={len(live_prices)}  "
                  f"joint_log_prob={log_prob.item():.3f}")
        print(header)
        print(f"  {'Asset':<6} {'Price':>10}  {'Action':>7}  {'Direction':<5}  "
              f"{'Confidence':>10}  {'μ':>7}  {'σ':>6}")
        print(f"  {'-'*6} {'-'*10}  {'-'*7}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*6}")
        for i, a in enumerate(self.assets):
            act  = action_list[i]
            mu_a = mu_list[i]
            sd_a = std_list[i]
            conf = abs(act)          # 0 = uncertain/hold, 1 = max confidence
            if act > HOLD_THRESH:
                direction = "BUY"
            elif act < -HOLD_THRESH:
                direction = "SELL"
            else:
                direction = "HOLD"
            price = live_prices.get(a, 0.0)
            print(f"  {a:<6} {price:>10.4f}  {act:>+7.4f}  {direction:<5}  "
                  f"{conf:>10.4f}  {mu_a:>+7.4f}  {sd_a:>6.4f}")

        # Current exposure summary
        exposure_str = "  ".join(
            f"{a}:{self.target_exposures.get(a, 0.0):+.3f}"
            for a in self.assets
        )
        print(f"  exposures: {exposure_str}")

        # ----------------------------------------------------------------
        # Action → order conversion
        #
        # target_exposures[a] ∈ [-1, +1] is a fraction of total equity.
        # Positive = long, negative = close/flatten (spot-only, no shorting).
        # Delta vs current tracked position drives the actual order size.
        #
        # Min trade: $1 USD notional (avoids API rejection for dust orders).
        # ----------------------------------------------------------------
        MIN_TRADE_USD = 5.0   # Kraken per-asset minimums; keep well above them

        # Fee constants — pulled from the broker so they stay in one place.
        # Kraken spot taker = 0.26 %; round-trip = 0.52 %.
        TAKER_FEE      = getattr(self.broker, "taker_fee", 0.0026)
        ROUND_TRIP     = 2.0 * TAKER_FEE          # min gross gain to break even
        FEE_BUFFER     = 2.0                       # require 2× round-trip to trade
        MIN_FEE_HURDLE = ROUND_TRIP * FEE_BUFFER   # ~1.04 % of notional

        # Rebalancing cooldown — 5 minutes between rounds.
        # 90s was too aggressive for a $100 account: 30 trades/hour limit
        # was hit within ~45 min, burning $2+ in fees.  At 5 min we get
        # ≤12 rounds/hour; with the 6 % drift gate and 0.60 min-confidence
        # filter that typically means 3–8 actual trades per hour.
        REBALANCE_COOLDOWN_SEC = 300
        _now = time.time()
        _last_rebal = getattr(self.broker, "_last_rebalance_time", 0.0)
        if _now - _last_rebal < REBALANCE_COOLDOWN_SEC:
            return   # too soon — skip this step
        if self.broker is not None:
            self.broker._last_rebalance_time = _now

        # Refresh balances from Kraken so deposits are picked up immediately
        if hasattr(self.broker, "fetch_live_balances"):
            self.broker.fetch_live_balances()

        live_balances = getattr(self.broker, "live_balances", {})

        # Anchor starting equity on first step (prices are loaded by now)
        if (hasattr(self.broker, "_starting_equity")
                and self.broker._starting_equity is None
                and hasattr(self.broker, "compute_total_equity")):
            anchor = self.broker.compute_total_equity()
            if anchor > 0:
                self.broker._starting_equity = anchor
                print(f"  [EQUITY ANCHOR] Starting equity = {anchor:.2f} (ZUSD + crypto at live prices)")

        # Total portfolio value for position sizing (ZUSD + all crypto at live prices)
        if hasattr(self.broker, "compute_total_equity"):
            equity = self.broker.compute_total_equity()
        else:
            equity = float(live_balances.get("ZUSD", "0") or 0)

        if equity <= 0:
            print("  [ORDER] No equity available — skipping order conversion")
            return

        # Free USD cash caps buy spending; sells are limited by actual holdings
        remaining_usd = float(live_balances.get("ZUSD", "0") or 0)

        # Sync spot_positions from live Kraken balances so sells reflect real holdings
        spot_positions = getattr(self.broker, "spot_positions", {})
        if hasattr(self.broker, "kraken_balance_keys"):
            for asset, bal_key in self.broker.kraken_balance_keys.items():
                qty = float(live_balances.get(bal_key, 0.0) or 0.0)
                if qty > 0:
                    spot_positions[asset] = qty

        # ----------------------------------------------------------------
        # Build a candidate list sorted by urgency (|delta_frac| desc) so
        # the most out-of-target positions get priority within each round.
        # Within the same round, execute SELLs before BUYs so the freed
        # cash is available for the subsequent buy orders.
        # ----------------------------------------------------------------
        MIN_DRIFT_FRAC  = 0.06   # 6 % drift gate (3 % was too active on $100 account)
        MIN_CONFIDENCE  = 0.60   # only act on reasonably high-conviction signals
        max_notional   = getattr(self.broker, "max_notional_per_asset", 50.0)

        candidates = []
        for a in self.assets:
            price = live_prices.get(a, 0.0)
            if price <= 0:
                continue
            current_units = spot_positions.get(a, 0.0)
            current_frac  = (current_units * price) / equity
            # Spot-only: negative target → flatten (0), no shorting
            target_frac   = max(0.0, self.target_exposures.get(a, 0.0))
            delta_frac    = target_frac - current_frac
            if abs(delta_frac) < MIN_DRIFT_FRAC:
                continue   # within tolerance — nothing to do

            # Confidence ∈ [0, 1] = |raw action|; scale notional cap so
            # high-conviction signals get larger position moves.
            confidence    = abs(self.target_exposures.get(a, 0.0))

            # Skip low-conviction signals — saves fees on noisy flips
            if confidence < MIN_CONFIDENCE:
                continue
            # Scale the cap: 50 % of max_notional at min confidence,
            # 100 % at confidence ≥ 0.7, linearly interpolated.
            conf_scale    = min(1.0, max(0.5, confidence / 0.7))
            scaled_cap    = max_notional * conf_scale

            delta_usd     = delta_frac * equity
            if delta_usd > 0:
                delta_usd = min(delta_usd, scaled_cap)
            else:
                delta_usd = max(delta_usd, -scaled_cap)

            notional_usd  = abs(delta_usd)
            delta_units   = notional_usd / price

            if delta_usd < 0:
                if current_units <= 0:
                    continue   # nothing to sell
                delta_units  = min(delta_units, current_units)
                notional_usd = delta_units * price

            if notional_usd < MIN_TRADE_USD:
                continue   # dust — skip

            candidates.append({
                "asset":        a,
                "side":         "sell" if delta_usd < 0 else "buy",
                "delta_frac":   delta_frac,
                "delta_usd":    delta_usd,
                "delta_units":  delta_units,
                "notional_usd": notional_usd,
                "current_frac": current_frac,
                "target_frac":  target_frac,
                "confidence":   confidence,
                "price":        price,
            })

        # Sort: SELLs first (frees cash), then BUYs; within each group
        # order by |delta_frac| descending (most urgent first).
        candidates.sort(key=lambda c: (0 if c["side"] == "sell" else 1,
                                       -abs(c["delta_frac"])))

        # Cap orders per round — prevents excess fee burn on a small account.
        # 4 orders × $0.13 avg fee = $0.52 max fee per 5-min round.
        MAX_ORDERS_PER_ROUND = 4
        orders_placed = 0

        for c in candidates:
            if orders_placed >= MAX_ORDERS_PER_ROUND:
                break
            a            = c["asset"]
            side         = c["side"]
            delta_usd    = c["delta_usd"]
            delta_units  = c["delta_units"]
            notional_usd = c["notional_usd"]
            target_frac  = c["target_frac"]
            current_frac = c["current_frac"]
            confidence   = c["confidence"]

            est_fee_usd = notional_usd * TAKER_FEE

            # Sell hurdle: gross position move must cover round-trip fee × buffer
            if side == "sell":
                gross_move_usd = abs(c["delta_frac"]) * equity
                min_required   = notional_usd * MIN_FEE_HURDLE
                if gross_move_usd < min_required:
                    print(f"  [ORDER] {a} sell skipped — gain "
                          f"({gross_move_usd:.2f}) < hurdle ({min_required:.2f})")
                    continue

            # Buy: full cost must fit in remaining cash
            if side == "buy":
                total_cost = notional_usd * (1.0 + TAKER_FEE)
                if total_cost > remaining_usd:
                    print(f"  [ORDER] {a} skipped — need ${total_cost:.2f},"
                          f" have ${remaining_usd:.2f}")
                    continue

            print(f"  [ORDER] {side.upper()} {delta_units:.6f} {a}"
                  f"  Δ={delta_usd:+.2f} USD  conf={confidence:.2f}"
                  f"  fee≈{est_fee_usd:.3f} USD"
                  f"  (target={target_frac:.3f}  current={current_frac:.3f}"
                  f"  cash_left={remaining_usd:.2f})")
            self.place_order(a, side, delta_units)
            orders_placed += 1

            if hasattr(self.broker, "cumulative_fees_usd"):
                self.broker.cumulative_fees_usd += est_fee_usd

            if side == "buy":
                remaining_usd -= notional_usd * (1.0 + TAKER_FEE)
            else:
                remaining_usd += notional_usd * (1.0 - TAKER_FEE)

        # ----------------------------------------------------------------
        # Futures overlay — runs after all spot orders are placed so the
        # spot core stays exactly as filled.  Uses the same target_exposures
        # already computed above; the broker applies regime-adaptive sizing,
        # vol targeting, funding adjustment and leverage caps before sending
        # any order to Kraken Futures.
        # ----------------------------------------------------------------
        if hasattr(self.broker, "run_futures_overlay"):
            self.broker.run_futures_overlay(self, live_prices)

    # ------------------------------------------------------------------
    # Order entry
    # ------------------------------------------------------------------

    def place_order(self, symbol, side, size):
        """
        Unified order entry point for the agent.

        In dry-run mode (default), logs the intended trade instead of
        sending it to the broker.  In live mode, delegates to
        broker.execute_trade() which runs the full safety harness.
        Every call (dry-run or live) is appended to the JSONL journal.

        Parameters
        ----------
        symbol : str   e.g. "SOL", "ETH"
        side   : str   "buy" or "sell"
        size   : float position size in base-asset units
        """
        if self.dry_run or self.broker is None:
            print(f"[DRY RUN] Would place order: {side} {size} {symbol}")
            record = {
                "status":    "dry_run",
                "symbol":    symbol,
                "side":      side,
                "size":      size,
                "timestamp": time.time(),
            }
            self.log_trade(record)
            return record

        result = self.broker.execute_trade(symbol, side, size)
        record = {
            "status":    "live",
            "symbol":    symbol,
            "side":      side,
            "size":      size,
            "result":    result,
            "timestamp": time.time(),
        }
        self.log_trade(record)
        return result

    def log_trade(self, record: dict, filename="trades.jsonl"):
        """
        Append a single trade record to a JSONL journal.
        Each line is one JSON object for easy real-time viewing.

        Usage
        -----
        tail -f trades.jsonl              # live monitoring
        pd.read_json("trades.jsonl", lines=True)  # analysis
        """
        if "timestamp" not in record:
            record["timestamp"] = time.time()

        try:
            with open(filename, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[JOURNAL] Failed to write trade record: {e}")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @property
    def num_actions(self):
        """For display compat: the 'actions' are now per-asset exposures."""
        return self.num_assets

    def reset_for_new_episode(self):
        """
        Reset per-episode fields.
        Network weights, replay buffer, and actor persist across episodes.
        """
        super().reset_for_new_episode()
        self.positions        = {a: 0.0 for a in self.assets}
        self.target_exposures = {a: 0.0 for a in self.assets}
        self.target_exposure  = 0.0
        self.last_mid_price   = None
        self.prev_state       = None
        self.prev_action      = None
        self.prev_equity      = None
        self.last_log_prob    = None
        self.prev_log_prob    = None
        self.eligibilities.clear()
        self.n_step_buffer.clear()

    # ------------------------------------------------------------------
    # State encoding (legacy — retained for reference / diagnostics)
    # ------------------------------------------------------------------

    def encode_state(self, market_state):
        """Legacy tabular encoding. Superseded by featurize_state()."""
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

        SOL-specific features (16)
        --------------------------
        A. price_bucket  – normalized 1-step price return, 3dp.
        B. vol_bucket    – current regime volatility parameter, 3dp.
        C. drift_bucket  – current drift, 3dp.
        D. inv_bucket    – agent's SOL exposure in [-1, +1] (float).
        E. regime        – regime string (converted to float in state_to_vector).
        F. m1            – 1-step momentum if MarketState provides it, else 0.
        G. m3            – 3-step momentum if MarketState provides it, else 0.
        H. imbalance     – order-flow imbalance if MarketState provides it, else 0.
        I. short_vol     – 5-step realized volatility, capped at 0.1.
        J. long_vol      – 20-step realized volatility, capped at 0.1.
        K. mom_5         – 5-bar price momentum, clipped to ±0.2.
        L. mom_20        – 20-bar price momentum, clipped to ±0.2.
        M. mom_50        – 50-bar price momentum, clipped to ±0.2.
        N. rolling_vol   – 20-bar avg synthetic volume, normalised /1000, capped 1.
        O. vol_imbalance – buy−sell volume, normalised /1000, clipped ±1.
        P. pressure      – vol_imbalance / total_volume, in (−1, +1).

        Cross-asset features (64 = 8 × 8 assets: SOL, XRP, LINK, ETH, BTC, XLM, HBAR, AVAX)
        -----------------------------------------------------------------------
        Per asset: price/1000, vol, mom_5/100, mom_20/100, mom_50/100,
                   rolling_vol/1000, vol_imbalance/1000, pressure.

        Crypto-regime features (7)
        --------------------------
        btc_dominance, eth_btc_ratio/100, sol_btc_strength/100,
        sol_eth_strength/100, altseason_index, vol_regime, liq_regime/1000.
        """
        prev_price = agent.last_mid_price
        if prev_price is not None and prev_price > 0:
            price_change = (market_state.mid_price - prev_price) / max(prev_price, 1e-6)
        else:
            price_change = 0.0
        price_bucket = round(price_change, 3)

        vol_bucket   = round(market_state.volatility, 3)
        drift_bucket = round(market_state.drift, 3)
        inv_bucket   = agent.position   # SOL exposure (kept in sync by broker)
        regime       = market_state.regime

        m1 = round(market_state.momentum_1, 3) if hasattr(market_state, "momentum_1") else 0
        m3 = round(market_state.momentum_3, 3) if hasattr(market_state, "momentum_3") else 0
        imbalance = (
            round(market_state.order_imbalance, 3)
            if hasattr(market_state, "order_imbalance") else 0
        )

        short_vol = round(min(getattr(market_state, "short_vol", 0.0), 0.1), 5)
        long_vol  = round(min(getattr(market_state, "long_vol",  0.0), 0.1), 5)

        mom_5  = max(min(getattr(market_state, "mom_5",  0.0), 0.2), -0.2)
        mom_20 = max(min(getattr(market_state, "mom_20", 0.0), 0.2), -0.2)
        mom_50 = max(min(getattr(market_state, "mom_50", 0.0), 0.2), -0.2)

        rolling_vol   = min(getattr(market_state, "rolling_vol",   0.0) / 1000.0, 1.0)
        vol_imbalance = max(min(getattr(market_state, "vol_imbalance", 0.0) / 1000.0, 1.0), -1.0)
        pressure      = max(min(getattr(market_state, "pressure",   0.0), 1.0), -1.0)

        features = [
            price_bucket, vol_bucket, drift_bucket, inv_bucket, regime,
            m1, m3, imbalance,
            short_vol, long_vol,
            mom_5, mom_20, mom_50,
            rolling_vol, vol_imbalance, pressure,
        ]

        for a in ["SOL", "XRP", "LINK", "ETH", "BTC", "XLM", "HBAR", "AVAX"]:
            features.extend([
                getattr(market_state, f"{a}_price",        0.0) / 1000.0,
                getattr(market_state, f"{a}_vol",          0.0),
                getattr(market_state, f"{a}_mom_5",        0.0) / 100.0,
                getattr(market_state, f"{a}_mom_20",       0.0) / 100.0,
                getattr(market_state, f"{a}_mom_50",       0.0) / 100.0,
                getattr(market_state, f"{a}_rolling_vol",  0.0) / 1000.0,
                getattr(market_state, f"{a}_vol_imbalance",0.0) / 1000.0,
                getattr(market_state, f"{a}_pressure",     0.0),
            ])

        features.extend([
            getattr(market_state, "btc_dominance",    0.0),
            getattr(market_state, "eth_btc_ratio",    0.0) / 100.0,
            getattr(market_state, "sol_btc_strength", 0.0) / 100.0,
            getattr(market_state, "sol_eth_strength", 0.0) / 100.0,
            getattr(market_state, "altseason_index",  0.0),
            getattr(market_state, "vol_regime",       0.0),
            getattr(market_state, "liq_regime",       0.0) / 1000.0,
        ])

        # Feature 64: macro_regime flag — +1 bull, 0 neutral, -1 bear.
        # Lets the network learn different behaviour under different
        # macro stances without re-training from scratch each time.
        features.append(float(getattr(market_state, "macro_regime", 0)))

        return tuple(features)

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
    # Value helpers — distributional V(s)
    # ------------------------------------------------------------------

    def _dist_to_value(self, logits):
        """
        Convert raw distribution logits to expected scalar value.

        logits  : [batch, num_atoms]  (raw network output)
        Returns : [batch]             (expected value E[Z] = Σ p(z)·z)
        """
        probs   = torch.softmax(logits, dim=-1)          # [batch, num_atoms]
        support = self.support.view(1, -1)               # [1, num_atoms]
        return (probs * support).sum(dim=-1)             # [batch]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def decide(self, market_state):
        """
        Sample a continuous portfolio action from the actor.

        Returns a tensor of shape (num_assets,) with target exposures in
        [-1, +1] for [SOL, XRP, LINK, ETH, BTC, XLM, HBAR, AVAX].

        Sets self.target_exposures (dict) and self.last_log_prob for the
        subsequent actor update.
        """
        state_tuple = self.featurize_state(market_state, self)
        self.last_mid_price = market_state.mid_price

        state_vec = self.state_to_vector(state_tuple)
        x = torch.tensor(state_vec, dtype=torch.float32,
                          device=self.device).unsqueeze(0)   # [1, feature_dim]

        self.actor.train()
        action, log_prob = self.actor.sample(x)   # [1, num_assets], [1]

        self.last_log_prob = log_prob              # store for actor update next step

        action_list = action.detach().squeeze(0).tolist()
        for i, a in enumerate(self.assets):
            self.target_exposures[a] = float(action_list[i])

        # Legacy compat: single SOL exposure target
        self.target_exposure = self.target_exposures["SOL"]

        return action.detach().squeeze(0)          # (num_assets,) tensor

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def compute_reward(self, raw_reward, market_state):
        """
        Shape the raw equity-change reward.

        1. Mean absolute portfolio exposure penalty — discourages large
           directional bets across all assets simultaneously.
        2. Volatility scaling — reduces reward magnitude in chaotic regimes.
        """
        if self.positions:
            mean_abs_exposure = sum(abs(v) for v in self.positions.values()) / len(self.positions)
        else:
            mean_abs_exposure = abs(self.position)
        inventory_penalty = self.INVENTORY_PENALTY_COEF * mean_abs_exposure
        reward = raw_reward - inventory_penalty
        vol_scale = 1.0 / (1.0 + market_state.volatility)
        return reward * vol_scale

    # ------------------------------------------------------------------
    # Eligibility traces (structural — not used in neural weight update)
    # ------------------------------------------------------------------

    def update_eligibilities(self, state, action):
        """Maintain accumulating eligibility traces (kept for structural consistency)."""
        decay = self.gamma * self.lambda_
        for key in list(self.eligibilities.keys()):
            self.eligibilities[key] *= decay
            if self.eligibilities[key] < 1e-6:
                del self.eligibilities[key]
        key = (state, id(action) if not isinstance(action, (int, str)) else action)
        self.eligibilities[key] = self.eligibilities.get(key, 0.0) + 1.0

    # ------------------------------------------------------------------
    # Experience replay (Prioritized, N-step)
    # ------------------------------------------------------------------

    def add_experience(self, prev_state, action, reward, new_state, done=False):
        """
        Accumulate a raw 1-step transition into the N-step buffer, then
        collapse and push to the replay buffer once N steps are available.

        N-step return: R_n = Σ_{k=0}^{N-1} γ^k · r_{t+k}
        The collapsed transition (s_0, a_0, R_n, s_n) is stored for the
        critic update; the actor uses online updates instead.

        Priority: new entries receive max priority so they are sampled at
        least once before being re-scored by actual critic loss.
        """
        self.n_step_buffer.append((prev_state, action, reward, new_state, done))

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        R = 0.0
        gamma_pow = 1.0
        for (_, _, r, _, d) in self.n_step_buffer:
            R += gamma_pow * r
            if d:
                break
            gamma_pow *= self.gamma

        s0, a0, _, _, _ = self.n_step_buffer[0]
        s_n = self.n_step_buffer[-1][3]

        transition = (s0, a0, R, s_n)
        max_prio = max((p for (_, p) in self.replay_buffer), default=1.0)
        self.replay_buffer.append((transition, max_prio))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

        if done:
            self.n_step_buffer.clear()

    def replay(self):
        """
        Sample a priority-weighted mini-batch and update the value network.

        Only the C51 critic is updated here (using N-step replay).
        The actor is updated online each step via update_actor() in simulation.
        """
        self._replay_call_count += 1
        if self._replay_call_count % self.replay_freq != 0:
            return
        if len(self.replay_buffer) < self.batch_size:
            return

        priorities = [p for (_, p) in self.replay_buffer]
        probs = [p ** self.prioritized_alpha for p in priorities]
        total = sum(probs)
        probs = [p / total for p in probs]

        indices = random.choices(range(len(self.replay_buffer)),
                                 weights=probs,
                                 k=self.batch_size)

        for idx in indices:
            transition, _ = self.replay_buffer[idx]
            prev_state, action, reward, new_state = transition

            td_error = self.update_value(prev_state, reward, new_state)

            new_prio = abs(td_error) + self.prioritized_epsilon
            self.replay_buffer[idx] = (transition, new_prio)
            self.replay_steps += 1

    # ------------------------------------------------------------------
    # Critic training step  V(s)  — C51 distributional
    # ------------------------------------------------------------------

    def update_value(self, prev_state, reward, new_state):
        """
        C51 distributional update for V(s).

        Algorithm
        ---------
        1. Build state tensors from feature tuples.
        2. Live value net forward (with gradient): logits [1, num_atoms].
        3. Target value net forward (no gradient): next_probs [num_atoms].
        4. C51 N-step Bellman projection:
               Tz_j = clamp(r + γ^n · z_j, v_min, v_max)
               Spread target_probs proportionally between floor/ceil atoms.
        5. Cross-entropy loss: −Σ_j target_j · log_prob_j
        6. Backprop → Adam step → resample noise → Polyak target update.
        7. Return loss.item() as PER priority signal.
        """
        s_vec  = self.state_to_vector(prev_state)
        s2_vec = self.state_to_vector(new_state)

        x  = torch.tensor(s_vec,  dtype=torch.float32, device=self.device).unsqueeze(0)
        x2 = torch.tensor(s2_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        r  = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # --- Live network forward ---
        logits   = self.value_net(x)                           # [1, num_atoms]
        log_probs = torch.log_softmax(logits, dim=-1)[0]       # [num_atoms]

        # --- Target V(s') ---
        with torch.no_grad():
            next_logits = self.target_value_net(x2)            # [1, num_atoms]
            next_probs  = torch.softmax(next_logits, dim=-1)[0]   # [num_atoms]

        # --- C51 distributional projection (N-step Bellman) ---
        gamma_n = self.gamma ** self.n_step
        Tz = r.unsqueeze(1) + gamma_n * self.support.view(1, -1)  # [1, num_atoms]
        Tz = Tz.clamp(self.v_min, self.v_max)

        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long().view(-1)
        u = b.ceil().long().view(-1)
        b = b.view(-1)

        target_dist = torch.zeros(self.num_atoms, device=self.device)
        target_dist.index_add_(0, l, next_probs * (u.float() - b))
        target_dist.index_add_(0, u, next_probs * (b - l.float()))

        # --- Cross-entropy loss ---
        loss = -(target_dist * log_probs).sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # --- Resample noise and soft-update target ---
        self.value_net.reset_noise()
        self.target_value_net.reset_noise()

        with torch.no_grad():
            for param, target_param in zip(
                self.value_net.parameters(), self.target_value_net.parameters()
            ):
                target_param.data.mul_(1.0 - self.target_update_tau)
                target_param.data.add_(self.target_update_tau * param.data)

        return loss.item()

    # ------------------------------------------------------------------
    # Actor training step — online advantage-weighted policy gradient
    # ------------------------------------------------------------------

    def update_actor(self, prev_state, reward, new_state):
        """
        One-step online actor update using the current value estimates.

        Advantage: A = r + γ · E[V(s')] − E[V(s)]
        Loss:      L = −A · log π(a|s)

        We *resample* a fresh action from the current actor on prev_state
        rather than using a stored log_prob from a previous forward pass.
        This avoids stale computation-graph errors caused by in-place
        parameter updates between the original sample and the backward call.
        """
        s_vec  = self.state_to_vector(prev_state)
        s2_vec = self.state_to_vector(new_state)

        x  = torch.tensor(s_vec,  dtype=torch.float32, device=self.device).unsqueeze(0)
        x2 = torch.tensor(s2_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        r  = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # Fresh sample → fresh computation graph anchored in current actor params
        self.actor.train()
        _action, log_prob = self.actor.sample(x)          # [1, num_assets], [1]

        with torch.no_grad():
            v_s  = self._dist_to_value(self.value_net(x))    # [1]
            v_s2 = self._dist_to_value(self.value_net(x2))   # [1]
            advantage = (r + self.gamma * v_s2 - v_s)        # [1]

        actor_loss = -(advantage * log_prob)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
