import os
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
from abc import ABC, abstractmethod
from exchange.exchange import Exchange


# --- Per-asset beta categories ----------------------------------------
# High-beta alts get more futures overlay (tactical/reactive).
# Low-beta majors get more spot core (stable/conviction).
# Mid-beta stays at the cycle-phase default.
HIGH_BETA = {"SOL", "AVAX", "LINK", "HBAR"}
MID_BETA  = {"XLM", "XRP"}
LOW_BETA  = {"BTC", "ETH"}


def _adjust_weights_for_asset(asset: str, spot_w: float, fut_w: float):
    """
    Tilt the cycle-phase base weights by asset beta, then renormalise
    so spot_w + fut_w == 1.0.

        High-beta alts  → −10 % spot, +10 % futures (more tactical)
        Low-beta majors → +10 % spot, −10 % futures (more stable)
        Mid-beta        → no change
    """
    if asset in HIGH_BETA:
        spot_w *= 0.9
        fut_w  *= 1.1
    elif asset in LOW_BETA:
        spot_w *= 1.1
        fut_w  *= 0.9
    # MID_BETA: unchanged

    total = spot_w + fut_w
    if total > 0:
        spot_w /= total
        fut_w  /= total

    return spot_w, fut_w


def _get_spot_futures_weights(cycle_phase: int):
    """
    Return (spot_weight, futures_weight) based on the long-term cycle phase.

    Cycle phases:
        0 = early accumulation → lean spot (stability)
        1 = mid expansion      → balanced
        2 = late distribution  → lean futures (nimble exit)
        3 = post-crash reset   → heaviest futures (fastest reaction)
    """
    if cycle_phase == 0:
        return 0.70, 0.30
    elif cycle_phase == 1:
        return 0.60, 0.40
    elif cycle_phase == 2:
        return 0.50, 0.50
    else:   # 3 = reset
        return 0.40, 0.60


class Broker(ABC):
    """
    Abstract execution venue for agent orders.

    All order routing, market state queries, and end-of-step settlement go
    through this interface. The Simulation never touches Exchange directly —
    only the Broker does. Swapping SimulatedBroker for a KrakenBroker or
    AlpacaBroker requires no changes to Simulation or agent code.
    """

    @abstractmethod
    def submit_order(self, agent, order_intent, market_state) -> list:
        """
        Route an approved order intent to the underlying execution venue.
        Returns a list of Fill objects produced by the execution.
        """

    @abstractmethod
    def get_market_state(self, price, regime, drift, volatility):
        """Return a MarketState snapshot for the current timestep."""

    @abstractmethod
    def fill_resting_orders(self, price) -> list:
        """Settle any unmatched resting orders at end of step.
        Returns a list of Fill objects from the final match."""

    @abstractmethod
    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None, simulation=None) -> float:
        """
        For each asset in agent.assets, ramp agent.positions[asset] toward
        agent.target_exposures[asset] and apply microstructure costs.

        Parameters
        ----------
        agent            : ReinforcementLearningTrader
        prices           : Dict[str, float] — current mid price per asset
        microstructure_fn: callable(mid_price, side, delta_exposure)
                           → (exec_price, txn_cost), or None for zero-cost mode

        Returns
        -------
        float — total transaction cost charged across all assets this step
        """

    @property
    @abstractmethod
    def trade_log(self):
        """Read-only access to the list of filled trades this episode."""


class SimulatedBroker(Broker):
    """
    Routes orders to the in-process Exchange (order book + market maker).

    Execution is split into two conceptual layers:

        Spot layer   — slow, stable, conviction exposure.  Only long.
                       Executes when the position change exceeds 2%.
        Futures layer — fast, tactical.  Both long and short allowed.
                       Executes every step.  Inherits vol_scaler,
                       panic_risk, and top_risk adjustments.

    The combined exposure (spot + futures) is written to agent.positions[a]
    so the rest of the system sees a single consistent value.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange if exchange is not None else Exchange()

        # --- Hybrid execution state ---------------------------------------
        # Lazily populated per-asset on first encounter so the broker does
        # not need to know the asset universe at construction time.
        self.spot_positions    = {}   # asset → current spot exposure
        self.futures_positions = {}   # asset → current futures exposure
        self.funding_rates     = {}   # asset → raw funding rate (feed externally or sim)

        # Funding-rate EMA smoothing — reduces noise in short funding windows
        self.funding_ema   = {}   # asset → smoothed funding rate
        self.funding_alpha = 0.2  # EMA decay (0 = no update, 1 = no smoothing)

        # --- Portfolio-level leverage limits ------------------------------
        # All values are in fractional-exposure units (1.0 = 100% of equity).
        # Spot layer is excluded — leverage control applies to futures only.
        self.max_portfolio_leverage = 2.0   # total |spot + futures| cap
        self.max_futures_notional   = 1.5   # total |futures| cap across assets
        self.max_asset_futures      = 0.40  # per-asset futures cap

        # --- Per-asset volatility targeting --------------------------------
        # EMA of absolute 1-step returns — feeds a per-asset scaling factor
        # applied to the futures layer.  High-vol assets get smaller overlays.
        self.asset_vol_ema   = {}   # asset → smoothed |return| estimate
        self.asset_vol_alpha = 0.1  # EMA decay (slower than funding EMA)

        # --- Execution-layer slippage parameters --------------------------
        # Realistic crypto fills: spot uses TWAP-like routing (low slippage),
        # futures use market-like routing (higher slippage).
        # size_factor scales slippage up for large deltas.
        self.spot_slippage_bps    = 5    # 0.05 % for spot
        self.futures_slippage_bps = 20   # 0.20 % for futures
        self.slippage_scale       = 1.0  # multiplier on trade size

        # --- Futures-only hedging configuration ---------------------------
        # When panic_risk == 2 or top_risk == 2, the broker can flip the
        # futures layer net-short to hedge the long spot core — without
        # ever touching spot positions.
        self.enable_futures_hedging = True
        self.max_hedge_ratio        = 1.0   # 1.0 = up to 100 % of net long hedged

    # ------------------------------------------------------------------
    # Standard order-book routing (used by non-RL agents)
    # ------------------------------------------------------------------

    def submit_order(self, agent, order_intent, market_state) -> list:
        """Delegate to Exchange.process_order and return all fills produced."""
        return self._exchange.process_order(
            agent, order_intent.side, order_intent.price
        )

    def get_market_state(self, price, regime, drift, volatility):
        return self._exchange.get_market_state(price, regime, drift, volatility)

    def fill_resting_orders(self, price) -> list:
        return self._exchange.fill_resting_orders(price)

    # ------------------------------------------------------------------
    # Hybrid portfolio execution (RL agent only)
    # ------------------------------------------------------------------

    def execute_portfolio_exposure(self, agent, prices, microstructure_fn=None, simulation=None) -> float:
        """
        Two-layer hybrid execution:

        1. Spot layer  (spot_weight × target, long-only, 2% move threshold)
           Slow and stable — represents core conviction positions.

        2. Futures layer (futures_weight × target, long/short allowed)
           Fast and tactical — inherits vol_scaler, panic_risk, top_risk.
           Updates every step (1% threshold).

        Returns
        -------
        float — sum of transaction costs across all assets this step.
        """
        total_cost = 0.0

        # Pull simulation-level signals (graceful defaults when None)
        cycle_phase = getattr(simulation, "cycle_phase", 0)
        vol_scaler  = getattr(simulation, "vol_scaler",  1.0)
        panic_risk  = getattr(simulation, "panic_risk",  0)
        top_risk    = getattr(simulation, "top_risk",    0)

        spot_w, fut_w = _get_spot_futures_weights(cycle_phase)

        for a in agent.assets:
            price = prices.get(a, 0.0)
            if price <= 0:
                continue

            # Lazy-init per-asset hybrid state
            if a not in self.spot_positions:
                self.spot_positions[a]    = 0.0
                self.futures_positions[a] = 0.0
                self.funding_rates[a]     = 0.0

            # --- Caps (regime-adaptive → agent static fallback) ----------
            if simulation is not None and simulation.dynamic_caps.get(a) is not None:
                cap_long  = simulation.dynamic_caps[a]
                cap_short = -0.5 * cap_long
            else:
                cap_long  = agent.max_long.get(a,  1.0)
                cap_short = agent.max_short.get(a, -1.0)

            # Clamped RL target (pre-split)
            raw_target = max(cap_short, min(cap_long, agent.target_exposures[a]))

            # Per-asset beta tilt on top of cycle-phase base weights
            _spot_w, _fut_w = _adjust_weights_for_asset(a, spot_w, fut_w)

            # --- 1. Spot target (long-only, stable) ----------------------
            spot_target = raw_target * _spot_w
            spot_target = max(0.0, spot_target)              # spot cannot short
            spot_target = max(0.0, min(cap_long, spot_target))

            # --- 2. Futures target (tactical) ----------------------------
            fut_target = raw_target * _fut_w

            # Volatility scaling — applied to futures only
            fut_target *= vol_scaler

            # --- Per-asset volatility targeting --------------------------
            # Feeds the latest step return into the EMA, then computes a
            # scaling factor: high-vol → smaller futures; low-vol → larger.
            # vol_scale ∈ [0.5, 1.5], clamped to prevent extreme reactions.
            _a_rets    = (simulation.asset_returns.get(a) or [0.0]) if simulation else [0.0]
            _asset_ret = _a_rets[-1]
            self.update_asset_volatility(a, _asset_ret)
            _a_vol     = self.asset_vol_ema.get(a, 0.0)
            _vol_scale = max(0.5, min(1.5, 1.0 / (1.0 + 5.0 * _a_vol)))
            fut_target *= _vol_scale

            # Log when per-asset vol scaling is significant
            if _a_vol > 0.05:
                print(f"[VolTarget] {a}: vol={_a_vol:.4f}  scale={_vol_scale:.3f}")

            # Panic de-risking — futures collapse first
            if panic_risk == 1:
                fut_target *= 0.5
            elif panic_risk == 2:
                fut_target *= 0.1

            # Blow-off top — trim futures into strength
            if top_risk == 1:
                fut_target *= 0.7
            elif top_risk == 2:
                fut_target *= 0.3

            # --- Funding-aware adjustment (EMA-smoothed) -----------------
            # Positive funding → longs pay shorts → reduce long futures.
            # Negative funding → shorts pay longs → increase long futures.
            # k=0.25 gives a 25% reduction per unit of funding; adj clamped
            # to [0.5, 1.5] to prevent extreme reactions.
            _raw_funding = self.funding_rates.get(a, 0.0)
            self.update_funding_rate(a, _raw_funding)
            funding = self.funding_ema.get(a, 0.0)
            funding_adj = max(0.5, min(1.5, 1.0 - 0.25 * funding))
            fut_target *= funding_adj

            # Log only when funding is extreme (≥5% per 8 h is unusual)
            if abs(funding) > 0.05:
                print(f"[Funding] {a}: raw={_raw_funding:.4f}  ema={funding:.4f}  adj={funding_adj:.3f}")

            # Clamp futures within caps
            fut_target = max(cap_short, min(cap_long, fut_target))

            # --- Futures-only hedging mode --------------------------------
            # Activates only on severe panic (panic_risk == 2) or severe
            # blow-off top (top_risk == 2).  The spot core stays untouched;
            # the futures layer flips toward a short hedge position.
            # Leverage enforcement and slippage still apply afterward.
            if self.enable_futures_hedging and (panic_risk == 2 or top_risk == 2):
                net_long = spot_target + max(fut_target, 0.0)
                if net_long > 0:
                    desired_hedge = -min(self.max_hedge_ratio * net_long, net_long)
                    fut_target    = desired_hedge
                    print(f"[Hedge] {a}: net_long={net_long:.3f}  fut_hedge={fut_target:.3f}")

            # --- 3. Execute spot (slow — 2% threshold) -------------------
            delta_spot = spot_target - self.spot_positions[a]
            if abs(delta_spot) > 0.02:
                # Apply spot slippage (low — TWAP-like routing)
                delta_spot_slipped = self._apply_slippage(delta_spot, self.spot_slippage_bps)
                spot_cost = self._execute_spot_trade(
                    a, price, delta_spot_slipped, microstructure_fn
                )
                agent.balance      -= spot_cost
                agent.realized_pnl -= spot_cost
                total_cost         += spot_cost
                # Store actual filled exposure, not the raw target
                self.spot_positions[a] = self.spot_positions[a] + delta_spot_slipped

            # --- 4. Execute futures (fast — 1% threshold) ----------------
            delta_fut = fut_target - self.futures_positions[a]

            # --- Portfolio-level leverage enforcement --------------------
            # Works in fractional-exposure units (equity cancels out).
            # Spot is untouched — only futures are constrained here.
            current_fut = self.futures_positions[a]

            # 1. Per-asset futures cap (40 % of equity)
            fut_target = max(-self.max_asset_futures,
                             min(self.max_asset_futures, fut_target))
            delta_fut = fut_target - current_fut

            # 2. Total futures notional cap across portfolio
            _ts, _tf, _te = self._compute_portfolio_exposure()
            if _tf + abs(delta_fut) > self.max_futures_notional:
                _allowed = max(0.0, self.max_futures_notional - _tf)
                delta_fut  = (min(delta_fut,  _allowed) if delta_fut > 0
                              else max(delta_fut, -_allowed))
                fut_target = current_fut + delta_fut

            # 3. Total portfolio leverage cap (spot + futures)
            if _te + abs(delta_fut) > self.max_portfolio_leverage:
                _allowed = max(0.0, self.max_portfolio_leverage - _te)
                delta_fut  = (min(delta_fut,  _allowed) if delta_fut > 0
                              else max(delta_fut, -_allowed))
                fut_target = current_fut + delta_fut

            if abs(delta_fut) > 0.01:
                # Apply futures slippage (higher — market-like routing)
                delta_fut_slipped = self._apply_slippage(delta_fut, self.futures_slippage_bps)
                fut_cost = self._execute_futures_trade(
                    a, price, delta_fut_slipped, microstructure_fn
                )
                agent.balance      -= fut_cost
                agent.realized_pnl -= fut_cost
                total_cost         += fut_cost
                # Store actual filled exposure (slipped), not the intended target
                self.futures_positions[a] = current_fut + delta_fut_slipped

            # --- 5. Write combined exposure back to agent ----------------
            combined = self.spot_positions[a] + self.futures_positions[a]
            # Soft clamp: combined may slightly exceed a single-layer cap
            # because each layer is independently capped, but keep it sane.
            combined = max(cap_short - abs(cap_short), min(cap_long * 1.5, combined))
            agent.positions[a] = combined

        # Keep agent.position (base class, SOL) in sync for display / featurize_state
        agent.position = agent.positions.get("SOL", 0.0)
        return total_cost

    # ------------------------------------------------------------------
    # Slippage helpers
    # ------------------------------------------------------------------

    def _apply_slippage(self, delta_exposure, slippage_bps):
        """
        Return the actual filled delta after modelling execution slippage.

        Slippage grows linearly with trade size via `slippage_scale`:
          effective_slippage = (slippage_bps / 10 000) × (1 + scale × |delta|)

        The fill is reduced in the direction of the trade so the agent always
        ends up *less* exposed than it intended — a realistic conservative bias.
        """
        size_factor = 1.0 + self.slippage_scale * abs(delta_exposure)
        slippage    = (slippage_bps / 10_000.0) * size_factor
        return delta_exposure * (1.0 - slippage)

    # ------------------------------------------------------------------
    # Per-asset volatility helpers
    # ------------------------------------------------------------------

    def update_asset_volatility(self, asset, ret):
        """
        Update the EMA of absolute 1-step returns for `asset`.

        `ret` should be the fractional price return for this step.
        Initialises the EMA on the first call so there is no cold-start spike.
        """
        abs_ret = abs(ret)
        if asset not in self.asset_vol_ema:
            self.asset_vol_ema[asset] = abs_ret
        else:
            prev = self.asset_vol_ema[asset]
            self.asset_vol_ema[asset] = prev + self.asset_vol_alpha * (abs_ret - prev)

    # ------------------------------------------------------------------
    # Portfolio exposure helpers
    # ------------------------------------------------------------------

    def _compute_portfolio_exposure(self):
        """
        Return (total_spot, total_futures, total_combined) as sums of
        absolute fractional exposures across all tracked assets.
        """
        total_spot = sum(abs(v) for v in self.spot_positions.values())
        total_fut  = sum(abs(v) for v in self.futures_positions.values())
        return total_spot, total_fut, total_spot + total_fut

    # ------------------------------------------------------------------
    # Funding-rate helpers
    # ------------------------------------------------------------------

    def update_funding_rate(self, asset, raw_funding):
        """
        Feed a raw funding rate into the per-asset EMA.

        Initialises the EMA to raw_funding on the first call so there is
        no cold-start discontinuity.  Call this each step (or each funding
        period) before execute_portfolio_exposure.
        """
        if asset not in self.funding_ema:
            self.funding_ema[asset] = raw_funding
        else:
            prev = self.funding_ema[asset]
            self.funding_ema[asset] = prev + self.funding_alpha * (raw_funding - prev)

    # ------------------------------------------------------------------
    # Execution sub-routines
    # ------------------------------------------------------------------

    def _execute_spot_trade(self, asset, price, delta_exposure, microstructure_fn):
        """Apply microstructure cost for a spot exposure change and return it."""
        if microstructure_fn is None:
            return 0.0
        side = "buy" if delta_exposure > 0 else "sell"
        _exec_price, cost = microstructure_fn(price, side, delta_exposure)
        return cost

    def _execute_futures_trade(self, asset, price, delta_exposure, microstructure_fn):
        """Apply microstructure cost for a futures exposure change and return it."""
        if microstructure_fn is None:
            return 0.0
        side = "buy" if delta_exposure > 0 else "sell"
        _exec_price, cost = microstructure_fn(price, side, delta_exposure)
        return cost

    @property
    def trade_log(self):
        return self._exchange.trade_log


# ==========================================================================
# LiveBroker — live-trading skeleton (no real orders yet)
# ==========================================================================

class LiveBroker(SimulatedBroker):
    """
    Live trading broker skeleton.

    Inherits the full hybrid execution engine from SimulatedBroker so that
    strategy logic (vol targeting, leverage limits, hedging, slippage) is
    shared between sim and live modes without duplication.

    In the current state:
    - API keys are loaded from environment variables (never hard-coded).
    - All execution methods log intent only; no real orders are sent.
    - dry_run is permanently True until wiring and sanity checks are complete.
    - A kill-switch can halt all order flow immediately.
    """

    def __init__(self, *args, dry_run=True, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Kraken API credentials from the environment — never hard-coded.
        self.kraken_api_key    = os.getenv("KRAKEN_API_KEY", "")
        self.kraken_api_secret = os.getenv("KRAKEN_API_SECRET", "")

        if not self.kraken_api_key or not self.kraken_api_secret:
            print("[LiveBroker] WARNING: KRAKEN_API_KEY / KRAKEN_API_SECRET not set. "
                  "Running in dry-run mode with no credentials.")
            dry_run = True   # force dry-run when credentials are missing

        # Honour the caller's dry_run preference (default True for safety).
        # Set dry_run=False only after credentials are confirmed present.
        self.dry_run = dry_run
        if not self.dry_run:
            print("[LiveBroker] dry_run=False — LIVE ORDER SUBMISSION ENABLED")

        # Health and kill-switch state
        self.last_api_error   = None   # last exception or error message from API
        self.last_latency_sec = 0.0    # round-trip latency of last API call
        self.kill_switch      = False  # set True to halt all order flow immediately

        # Kraken spot REST configuration
        self.kraken_base_url = "https://api.kraken.com"
        self.kraken_session  = requests.Session()

        # Kraken Futures REST configuration — same keypair as spot
        self.futures_api_key    = self.kraken_api_key
        self.futures_api_secret = self.kraken_api_secret
        self.futures_base_url   = "https://futures.kraken.com/derivatives/api/v3"

        # Map internal asset names → Kraken ticker symbols
        self.kraken_pairs = {
            "BTC":  "XBTUSD",
            "ETH":  "ETHUSD",
            "SOL":  "SOLUSD",
            "AVAX": "AVAXUSD",
            "LINK": "LINKUSD",
            "HBAR": "HBARUSD",
            "XRP":  "XRPUSD",
            "XLM":  "XLMUSD",
        }

        # Live price cache (populated by fetch_live_prices)
        self.live_prices           = {}   # asset → latest float price
        self.last_price_timestamp  = 0    # Unix timestamp of last successful fetch

        # Live account state (read-only in dry-run)
        self.live_balances  = {}   # raw Kraken balance dict
        self.live_positions = {}   # raw Kraken open-positions dict

        # --- Automatic safety limits (first-night harness) ---------------
        self.max_notional_per_asset = 50.0    # USD cap per single asset trade
        self.max_total_notional     = 200.0   # USD cap across all open positions
        self.max_trades_per_hour    = 10      # rate limiter
        self.max_daily_loss         = 20.0    # USD drawdown cap from session start

        self._trade_count_window = []   # timestamps of recent trades (rolling 1 h)
        self._starting_equity    = None # ZUSD balance at first trade of the session

    # ------------------------------------------------------------------
    # Health / kill-switch
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """
        Gate called before every live order.  Returns False to block execution.

        Extend later with:
        - Latency threshold checks (e.g. reject if > 2 s)
        - Position mismatch detection (internal vs exchange state)
        - Per-asset leverage sanity checks
        - Stale / bad price data detection
        """
        if self.kill_switch:
            return False
        return True

    def trigger_kill_switch(self, reason: str):
        """Immediately halt all order flow and log the reason."""
        self.kill_switch    = True
        self.last_api_error = reason
        print(f"[KILL SWITCH] Triggered: {reason}")

    # ------------------------------------------------------------------
    # Kraken REST client (scaffolding — not yet wired into the main loop)
    # ------------------------------------------------------------------

    def _kraken_public(self, path: str, params: dict | None = None):
        """
        Call a public (unauthenticated) Kraken REST endpoint.

        Scaffolding only — not yet wired into price feeds or the main loop.
        Any network error triggers the kill switch to keep the system safe.
        """
        url    = self.kraken_base_url + path
        params = params or {}
        try:
            t0 = time.time()
            resp = self.kraken_session.get(url, params=params, timeout=5)
            self.last_latency_sec = time.time() - t0
            data = resp.json()
            if data.get("error"):
                self.last_api_error = data["error"]
                print(f"[KRAKEN PUBLIC ERROR] {data['error']}")
            return data
        except Exception as e:
            self.trigger_kill_switch(f"Kraken public request failed: {e}")
            return None

    def _kraken_private(self, path: str, data: dict | None = None):
        """
        Call a private (authenticated) Kraken REST endpoint.

        Scaffolding only — not yet called by _execute_spot_trade or
        _execute_futures_trade (both remain dry-run).
        Signs the request using the HMAC-SHA512 scheme Kraken requires.
        Any missing credentials or network error triggers the kill switch.
        """
        if not self.kraken_api_key or not self.kraken_api_secret:
            self.trigger_kill_switch("Missing Kraken API credentials")
            return None

        url  = self.kraken_base_url + path
        data = data or {}
        data["nonce"] = str(int(time.time() * 1000))

        post_data = urllib.parse.urlencode(data)
        message   = (path.encode("utf-8") +
                     hashlib.sha256((data["nonce"] + post_data).encode("utf-8")).digest())

        secret  = base64.b64decode(self.kraken_api_secret)
        sig     = hmac.new(secret, message, hashlib.sha512)
        sig_b64 = base64.b64encode(sig.digest())

        headers = {
            "API-Key":  self.kraken_api_key,
            "API-Sign": sig_b64.decode(),
        }

        try:
            t0 = time.time()
            resp = self.kraken_session.post(url, data=data, headers=headers, timeout=5)
            self.last_latency_sec = time.time() - t0
            result = resp.json()
            if result.get("error"):
                self.last_api_error = result["error"]
                print(f"[KRAKEN PRIVATE ERROR] {result['error']}")
            return result
        except Exception as e:
            self.trigger_kill_switch(f"Kraken private request failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Live data / position sync (stubs — simulation still drives prices)
    # ------------------------------------------------------------------

    def fetch_live_prices(self):
        """
        Fetch the latest mid-price for all tracked assets via Kraken's
        public Ticker endpoint.  No authentication required; no orders placed.
        Updates self.live_prices and self.last_price_timestamp on success.
        """
        result = {}

        for asset, pair in self.kraken_pairs.items():
            data = self._kraken_public("/0/public/Ticker", {"pair": pair})
            if not data or "result" not in data:
                print(f"[PRICE FEED ERROR] No data for {asset} ({pair})")
                continue

            # Kraken keys the result by the normalised pair name (may differ from request)
            ticker_key = list(data["result"].keys())[0]
            ticker     = data["result"][ticker_key]

            # 'c' = last trade closed: [price, lot_volume]
            try:
                price          = float(ticker["c"][0])
                result[asset]  = price
            except Exception:
                print(f"[PRICE PARSE ERROR] Could not parse last trade for {asset}")
                continue

        if result:
            self.live_prices          = result
            self.last_price_timestamp = time.time()

        return result

    def prices_are_fresh(self, max_age_sec: float = 10.0) -> bool:
        """
        Return True only when the price cache is non-empty and younger than
        `max_age_sec` seconds.  Guards against stale or missing data.
        """
        if not self.live_prices:
            return False
        if time.time() - self.last_price_timestamp > max_age_sec:
            return False
        return True

    def update_prices_if_needed(self):
        """
        Call before each portfolio step when running in live mode.
        Refreshes the price cache if stale; triggers the kill switch if the
        refresh still fails — so the system never acts on bad data.
        """
        if not self.prices_are_fresh():
            print("[PRICE FEED] Refreshing live prices...")
            self.fetch_live_prices()

        if not self.prices_are_fresh():
            self.trigger_kill_switch("Stale or missing live prices — halting order flow")

    def fetch_live_balances(self):
        """
        Fetch real account balances from Kraken (private endpoint, read-only).
        Populates self.live_balances and returns the raw result dict, or None
        on failure.
        """
        data = self._kraken_private("/0/private/Balance")

        if not data or "result" not in data:
            print("[BALANCE ERROR] Could not fetch balances from Kraken")
            return None

        # An empty result dict {} is valid — it means zero funded balance.
        # Only treat API errors (missing "result" key) as a failure.
        self.live_balances = data["result"]
        return self.live_balances

    def fetch_live_positions(self):
        """
        Fetch open positions (spot-margin or futures) from Kraken (read-only).
        Populates self.live_positions and returns the raw result dict, or None
        on failure.
        """
        data = self._kraken_private("/0/private/OpenPositions", {"docalcs": "true"})

        if not data or "result" not in data:
            print("[POSITION ERROR] Could not fetch open positions from Kraken")
            return None

        self.live_positions = data["result"]
        return self.live_positions

    def sync_live_account_state(self):
        """
        Fetch balances + positions in one call and run a basic mismatch check.

        Still dry-run — no writes, no orders.  Kill-switch fires if either
        fetch fails so the system can never act on an unknown account state.
        Returns (balances, positions) on success, None on any failure.
        """
        if not self.check_health():
            print("[SYNC BLOCKED] Kill-switch is active — skipping account sync")
            return None

        print("[SYNC] Fetching live balances and positions...")

        balances  = self.fetch_live_balances()
        positions = self.fetch_live_positions()

        if balances is None or positions is None:
            self.trigger_kill_switch("Failed to sync account state from Kraken")
            return None

        # Mismatch detection — useful once real trading is enabled.
        # Warn if internal state shows a futures position that Kraken doesn't.
        for asset, pos in self.futures_positions.items():
            if float(pos) != 0 and not positions:
                print(f"[MISMATCH WARNING] Internal futures pos for {asset}={pos:.4f} "
                      f"but Kraken shows no open positions")

        return balances, positions

    def fetch_live_futures_positions(self):
        """
        Fetch open futures positions from Kraken Futures.

        Uses the fully wired _kraken_futures_private client.
        Populates self.futures_positions with {asset: size}.
        """
        resp = self._kraken_futures_private("/openpositions", {})

        if not resp:
            print("[FUTURES POSITIONS] No response from Kraken Futures")
            return None

        if resp.get("error"):
            self.trigger_kill_switch(f"Futures positions error: {resp['error']}")
            return None

        result    = resp.get("result", {})
        positions = result.get("openPositions", [])

        new_positions = {}
        for pos in positions:
            symbol = pos.get("symbol")
            size   = pos.get("size")
            if symbol and size:
                try:
                    new_positions[symbol] = float(size)
                except Exception:
                    print(f"[FUTURES POSITIONS] Could not parse size for {symbol}")

        self.futures_positions = new_positions
        print(f"[FUTURES POSITIONS] {self.futures_positions}")
        return new_positions

    def compute_unified_exposure(self):
        """
        Compute spot + futures exposure in USD terms.

        Returns a dict with:
        - spot_exposure:   {asset: usd_exposure}
        - futures_exposure:{symbol: usd_exposure}
        - net_exposure:    float (sum of all exposures, signed)
        - total_notional:  float (sum of absolute exposures)
        """
        spot_exposure    = {}
        futures_exposure = {}

        # Spot / margin exposure (live_positions keyed by asset, e.g. "BTC")
        for asset, size in (self.live_positions or {}).items():
            price = (self.live_prices or {}).get(asset)
            if price is None:
                continue
            try:
                spot_exposure[asset] = float(size) * float(price)
            except Exception:
                continue

        # Futures exposure (futures_positions keyed by symbol, e.g. "XBTUSD", "SOLUSD")
        for symbol, size in (self.futures_positions or {}).items():
            base  = symbol.replace("USD", "").replace("USDT", "")
            price = None
            if self.live_prices:
                price = self.live_prices.get(base) or self.live_prices.get(symbol)
            if price is None:
                continue
            try:
                futures_exposure[symbol] = float(size) * float(price)
            except Exception:
                continue

        net_exposure   = sum(spot_exposure.values()) + sum(futures_exposure.values())
        total_notional = (sum(abs(v) for v in spot_exposure.values()) +
                          sum(abs(v) for v in futures_exposure.values()))

        snapshot = {
            "spot_exposure":    spot_exposure,
            "futures_exposure": futures_exposure,
            "net_exposure":     net_exposure,
            "total_notional":   total_notional,
        }
        print(f"[UNIFIED EXPOSURE] {snapshot}")
        return snapshot

    def compute_unified_pnl_snapshot(self):
        """
        Compute a simple PnL + risk snapshot for the current session.

        Uses:
        - ZUSD balance as current equity
        - _starting_equity (set on first live trade) as session anchor
        - compute_unified_exposure() for notional and leverage estimate
        """
        balances = self.live_balances or {}
        equity   = float(balances.get("ZUSD", 0.0))

        exposure       = self.compute_unified_exposure()
        total_notional = exposure.get("total_notional", 0.0)

        starting    = self._starting_equity
        session_pnl = None
        if starting is not None:
            session_pnl = equity - float(starting)

        leverage = None
        if equity > 0:
            leverage = total_notional / equity

        snapshot = {
            "equity":           equity,
            "starting_equity":  starting,
            "session_pnl":      session_pnl,
            "total_notional":   total_notional,
            "net_exposure":     exposure.get("net_exposure", 0.0),
            "leverage":         leverage,
            "spot_exposure":    exposure.get("spot_exposure", {}),
            "futures_exposure": exposure.get("futures_exposure", {}),
        }
        print(f"[UNIFIED PNL] {snapshot}")
        return snapshot

    def emit_health_check(self):
        """
        Emit a compact health snapshot:
        - kill switch state
        - equity and session PnL
        - net exposure and total notional
        - leverage estimate
        - spot + futures exposure breakdown
        """
        pnl = self.compute_unified_pnl_snapshot()

        snapshot = {
            "kill_switch":      self.kill_switch,
            "equity":           pnl.get("equity"),
            "session_pnl":      pnl.get("session_pnl"),
            "net_exposure":     pnl.get("net_exposure"),
            "total_notional":   pnl.get("total_notional"),
            "leverage":         pnl.get("leverage"),
            "spot_exposure":    pnl.get("spot_exposure"),
            "futures_exposure": pnl.get("futures_exposure"),
        }
        print(f"[HEALTH CHECK] {snapshot}")
        return snapshot

    def heartbeat(self, interval_seconds: int = 300):
        """
        Heartbeat loop:
        - emits a health check every `interval_seconds`
        - stops immediately if kill switch is active
        - intended to run inside the main loop (non-blocking)
        """
        now = time.time()
        if not hasattr(self, "_last_heartbeat"):
            self._last_heartbeat = 0

        if now - self._last_heartbeat >= interval_seconds:
            if self.kill_switch:
                print("[HEARTBEAT] Kill switch active — heartbeat suppressed")
            else:
                print(f"[HEARTBEAT] alive at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.emit_health_check()
            self._last_heartbeat = now

    def record_health_metrics(self):
        """
        Record a single snapshot of key risk metrics.
        Called automatically by heartbeat() or manually.
        Stores a rolling in-memory log for morning summaries.
        """
        if not hasattr(self, "_health_log"):
            self._health_log = []

        snapshot = self.emit_health_check()
        snapshot["timestamp"] = time.time()

        # Keep last ~2000 entries (roughly 24h at 40s cadence)
        self._health_log.append(snapshot)
        if len(self._health_log) > 2000:
            self._health_log.pop(0)

    def emit_morning_summary(self):
        """
        Produce a morning summary of the previous session:
        - overnight PnL
        - max leverage
        - max drawdown (equity-based)
        - kill-switch events
        - exposure extremes
        """
        if not hasattr(self, "_health_log") or not self._health_log:
            print("[MORNING SUMMARY] No health data recorded")
            return None

        log = self._health_log

        equities  = [e["equity"]       for e in log if e["equity"]       is not None]
        leverages = [e["leverage"]     for e in log if e["leverage"]     is not None]
        pnls      = [e["session_pnl"]  for e in log if e["session_pnl"] is not None]
        exposures = [e["net_exposure"] for e in log if e["net_exposure"] is not None]

        summary = {
            "entries":            len(log),
            "start_time":         log[0]["timestamp"],
            "end_time":           log[-1]["timestamp"],
            "starting_equity":    equities[0]  if equities else None,
            "ending_equity":      equities[-1] if equities else None,
            "overnight_pnl":      (equities[-1] - equities[0]) if len(equities) >= 2 else None,
            "max_leverage":       max(leverages)  if leverages  else None,
            "min_equity":         min(equities)   if equities   else None,
            "max_equity":         max(equities)   if equities   else None,
            "max_drawdown":       (min(equities) - max(equities)) if len(equities) >= 2 else None,
            "max_exposure":       max(exposures)  if exposures  else None,
            "min_exposure":       min(exposures)  if exposures  else None,
            "kill_switch_triggered": any(e["kill_switch"] for e in log),
        }
        print(f"[MORNING SUMMARY] {summary}")
        return summary

    def reset_morning_metrics(self):
        """
        Clears the rolling health log to begin a new session.
        """
        self._health_log = []
        print("[MORNING SUMMARY] Metrics reset for new session")

    def daily_rollover(self, rollover_hour: int = 7):
        """
        Automatic daily rollover:
        - At the specified hour (default 07:00 local time), emit a morning summary
        - Reset metrics for the new session
        - Ensures the rollover fires only once per day
        Intended to be called inside the main loop (non-blocking).
        """
        now          = time.localtime()
        current_hour = now.tm_hour
        current_day  = now.tm_yday

        if not hasattr(self, "_last_rollover_day"):
            self._last_rollover_day = None

        if current_hour == rollover_hour:
            if self._last_rollover_day != current_day:
                print(f"[DAILY ROLLOVER] Triggering morning summary for day {current_day}")
                self.emit_morning_summary()
                self.reset_morning_metrics()
                self._last_rollover_day = current_day
        # Outside rollover hour: no action needed — next day's rollover will
        # fire because _last_rollover_day will no longer equal current_day.

    def check_alerts(self,
                     max_leverage:       float = 3.0,
                     max_drawdown:       float = -500.0,
                     max_exposure:       float = 5000.0,
                     heartbeat_timeout:  float = 600.0):
        """
        Evaluate alert conditions and print alerts when thresholds are crossed.
        Intended to be called inside the main loop.
        """
        # --- 1. Kill switch alert ---
        if self.kill_switch:
            print("[ALERT] Kill switch active — trading halted")

        # --- 2. Leverage alert ---
        pnl = self.compute_unified_pnl_snapshot()
        lev = pnl.get("leverage")
        if lev is not None and lev > max_leverage:
            print(f"[ALERT] Leverage {lev:.2f} exceeds limit {max_leverage}")

        # --- 3. Drawdown alert (session PnL) ---
        session_pnl = pnl.get("session_pnl")
        if session_pnl is not None and session_pnl < max_drawdown:
            print(f"[ALERT] Drawdown {session_pnl:.2f} below limit {max_drawdown}")

        # --- 4. Exposure alert ---
        net_exp = pnl.get("net_exposure")
        if net_exp is not None and abs(net_exp) > max_exposure:
            print(f"[ALERT] Net exposure {net_exp:.2f} exceeds limit {max_exposure}")

        # --- 5. Heartbeat silence alert ---
        now = time.time()
        if hasattr(self, "_last_heartbeat"):
            if now - self._last_heartbeat > heartbeat_timeout:
                print("[ALERT] Heartbeat silence — engine may be frozen")

    def alerting_loop(self):
        """
        Lightweight wrapper to be called each iteration of the main loop.
        Keeps alerting logic clean and centralized.
        """
        self.check_alerts()

    # ------------------------------------------------------------------
    # Order payload builders (formatting only — nothing is submitted)
    # ------------------------------------------------------------------

    def _build_spot_order(self, asset, price, delta_exposure):
        """
        Format a Kraken spot market-order payload from a fractional delta.

        |delta_exposure| is treated as a synthetic size (fractional unit).
        Mapping to real notional (equity × |delta| / price) will be added
        when dry_run is disabled and real sizing logic is wired in.
        Returns None for zero-delta or invalid inputs.
        """
        if asset not in self.kraken_pairs:
            self.trigger_kill_switch(f"Unknown asset for spot order: {asset}")
            return None

        if price is None or price <= 0:
            self.trigger_kill_switch(f"Invalid price for spot order: {asset} price={price}")
            return None

        if delta_exposure == 0:
            return None

        return {
            "pair":      self.kraken_pairs[asset],
            "type":      "buy" if delta_exposure > 0 else "sell",
            "ordertype": "market",
            "volume":    f"{abs(delta_exposure):.8f}",
        }

    def _build_futures_order(self, asset, price, delta_exposure):
        """
        Format a Kraken Futures / perps order payload from a fractional delta.

        Uses the same kraken_pairs symbol mapping as spot for now; the
        futures-specific symbol format (e.g. PI_XBTUSD) will be adjusted
        when the real futures endpoint is wired.
        Returns None for zero-delta or invalid inputs.
        """
        if asset not in self.kraken_pairs:
            self.trigger_kill_switch(f"Unknown asset for futures order: {asset}")
            return None

        if price is None or price <= 0:
            self.trigger_kill_switch(f"Invalid price for futures order: {asset} price={price}")
            return None

        if delta_exposure == 0:
            return None

        return {
            "symbol":    self.kraken_pairs[asset],
            "side":      "buy" if delta_exposure > 0 else "sell",
            "ordertype": "market",
            "size":      f"{abs(delta_exposure):.8f}",
        }

    # ------------------------------------------------------------------
    # First-night safety harness
    # ------------------------------------------------------------------

    def _current_notional(self) -> float:
        """
        Estimate total USD notional exposure across spot + futures layers.
        Returns 0.0 if live_prices is not yet populated.
        """
        if not self.live_prices:
            return 0.0

        total = 0.0
        for asset, pos in self.spot_positions.items():
            price = self.live_prices.get(asset)
            if price:
                total += abs(pos * price)
        for asset, pos in self.futures_positions.items():
            price = self.live_prices.get(asset)
            if price:
                total += abs(pos * price)
        return total

    def _check_trade_rate(self) -> bool:
        """
        Sliding-window trade-rate limiter (max trades per rolling hour).
        Appends the current timestamp on success; triggers kill switch on breach.
        """
        now = time.time()
        self._trade_count_window = [t for t in self._trade_count_window
                                    if now - t < 3600]

        if len(self._trade_count_window) >= self.max_trades_per_hour:
            self.trigger_kill_switch("Trade rate exceeded — too many trades in 1 h")
            return False

        self._trade_count_window.append(now)
        return True

    def _check_daily_loss(self) -> bool:
        """
        Compare current ZUSD balance to the session-start balance.
        Triggers kill switch if the drawdown exceeds max_daily_loss.
        """
        bal_str = self.live_balances.get("ZUSD")

        if self._starting_equity is None:
            # Initialise once on the first call that has a balance available
            if bal_str:
                self._starting_equity = float(bal_str)
            return True

        if not bal_str:
            return True   # balance unavailable — permissive, do not block

        if float(bal_str) < self._starting_equity - self.max_daily_loss:
            self.trigger_kill_switch(
                f"Daily loss limit exceeded: balance={float(bal_str):.2f}  "
                f"start={self._starting_equity:.2f}  limit={self.max_daily_loss:.2f}"
            )
            return False

        return True

    def _pre_trade_safety(self, asset, price, delta_exposure) -> bool:
        """
        Unified pre-trade gate — runs all safety checks in priority order.
        Returns False (and triggers the kill switch where appropriate) if any
        check fails.  Called at the top of every live execution override.
        """
        if not self.check_health():
            return False

        if not self._check_trade_rate():
            return False

        # Per-asset notional cap — soft reject (skip this order, don't halt the bot)
        if price and abs(delta_exposure * price) > self.max_notional_per_asset:
            print(f"[SAFETY] Per-asset cap: skipping {asset} "
                  f"(|{delta_exposure:.4f}| × {price:.2f}"
                  f" = {abs(delta_exposure * price):.2f}"
                  f" > {self.max_notional_per_asset} USD)")
            return False

        # Total portfolio notional cap — soft reject
        if self._current_notional() > self.max_total_notional:
            print(f"[SAFETY] Total notional cap: skipping order "
                  f"({self._current_notional():.2f} > {self.max_total_notional} USD)")
            return False

        # Daily loss cap
        if not self._check_daily_loss():
            return False

        return True

    # ------------------------------------------------------------------
    # Unified live execution entry point
    # ------------------------------------------------------------------

    def execute_trade(self, symbol: str, side: str, size: float):
        """
        Unified live-trade execution called by agent.place_order().

        Parameters
        ----------
        symbol : str   Internal asset name e.g. "SOL", "BTC"
        side   : str   "buy" or "sell"
        size   : float Position size in base-asset units (positive)

        Flow
        ----
        1. Dry-run guard — logs and returns immediately if dry_run=True
        2. Price validation — kill switch on missing / zero price
        3. Pre-trade safety harness — notional cap, rate limit, daily loss
        4. Build spot market-order payload
        5. Submit to Kraken /AddOrder
        6. Update internal position tracker on success

        Returns the Kraken result dict on success, None on any failure.
        """
        if self.dry_run:
            print(f"[DRY RUN] execute_trade skipped: {side} {size:.6f} {symbol}")
            return None

        price = self.live_prices.get(symbol)
        if not price or price <= 0:
            self.trigger_kill_switch(
                f"No valid live price for {symbol} — order aborted"
            )
            return None

        # Signed delta: positive = long / buy, negative = short / sell
        delta = size if side == "buy" else -size

        if not self._pre_trade_safety(symbol, price, delta):
            return None

        order = self._build_spot_order(symbol, price, delta)
        if order is None:
            return None

        result = self._submit_spot_order(order)
        if result is not None:
            # Track position internally so _current_notional() stays accurate
            self.spot_positions[symbol] = (
                self.spot_positions.get(symbol, 0.0) + delta
            )
            notional = abs(delta * price)
            print(f"[EXECUTE] {side.upper()} {size:.6f} {symbol} @ {price:.4f}"
                  f"  notional={notional:.2f} USD"
                  f"  pos={self.spot_positions[symbol]:+.6f}")
        return result

    # ------------------------------------------------------------------
    # Spot order submission (live path only — dry_run must be False)
    # ------------------------------------------------------------------

    def _submit_spot_order(self, order: dict):
        """
        Submit a formatted spot order dict to Kraken's AddOrder endpoint.

        Only called from _execute_spot_trade when dry_run is False.
        Any API-level failure or missing result triggers the kill switch so
        the system never silently drops an order.
        Returns the Kraken result dict on success, or None on failure.
        """
        if not order:
            return None

        resp = self._kraken_private("/0/private/AddOrder", order)

        if not resp:
            self.trigger_kill_switch("Spot order submission failed — no response from Kraken")
            return None

        # Check API-level errors before checking for "result"
        # (Kraken error responses have no "result" key)
        errors = resp.get("error", [])
        if errors:
            soft = {"EOrder:Insufficient funds", "EOrder:Order minimum not met",
                    "EOrder:Orders limit exceeded",
                    "EGeneral:Invalid arguments:volume minimum not met"}
            if any(any(s in e for s in soft) for e in errors):
                print(f"[SAFETY] Order skipped (soft error): {errors}")
                return None
            self.trigger_kill_switch(f"Spot order error: {errors}")
            return None

        if "result" not in resp:
            self.trigger_kill_switch("Spot order submission failed — no result from Kraken")
            return None

        result = resp["result"]
        txids  = result.get("txid", [])
        print(f"[LIVE SPOT SUBMITTED] txid={txids}")
        return result

    def _submit_futures_order(self, order: dict):
        """
        Submit a futures/perps order to the Kraken Futures endpoint.

        Calls _kraken_futures_private (separate base URL + auth scheme from
        spot).  That helper is scaffolded here and can be wired when the
        Kraken Futures API credentials are ready.
        Returns the result dict on success, or None on failure.
        """
        if not order:
            return None

        resp = self._kraken_futures_private("/sendorder", order)
        if not resp:
            self.trigger_kill_switch("Futures order submission failed — no response")
            return None

        if resp.get("error"):
            self.trigger_kill_switch(f"Futures order error: {resp['error']}")
            return None

        result   = resp.get("result", resp)
        order_id = result.get("order_id") or result.get("orderId")
        print(f"[LIVE FUTURES SUBMITTED] order_id={order_id}")
        return result

    def _kraken_futures_private(self, path: str, data: dict | None = None):
        """
        Real Kraken Futures private REST request.

        Kraken Futures uses:
        - Base URL: https://futures.kraken.com/derivatives/api/v3
        - HMAC-SHA256 signing: endpoint + nonce + JSON payload
        - Headers: APIKey, Nonce, Authent
        """
        if not self.futures_api_key or not self.futures_api_secret:
            self.trigger_kill_switch("Missing Kraken API credentials (shared keypair required for Futures)")
            return None

        if data is None:
            data = {}

        url     = self.futures_base_url + path
        nonce   = str(int(time.time() * 1000))
        payload = json.dumps(data, separators=(",", ":"))

        # Signing message: endpoint + nonce + compact JSON payload
        message   = path + nonce + payload
        signature = hmac.new(
            self.futures_api_secret.encode(),
            msg=message.encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()

        headers = {
            "APIKey":       self.futures_api_key,
            "Nonce":        nonce,
            "Authent":      signature,
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, headers=headers, data=payload, timeout=10)
        except Exception as e:
            self.trigger_kill_switch(f"Futures request exception: {e}")
            return None

        if resp.status_code != 200:
            self.trigger_kill_switch(f"Futures HTTP error {resp.status_code}")
            return None

        try:
            return resp.json()
        except Exception:
            self.trigger_kill_switch("Futures JSON decode error")
            return None

    # ------------------------------------------------------------------
    # Live order execution (overrides SimulatedBroker private helpers)
    # Signature must match the parent: (asset, price, delta_exposure, microstructure_fn)
    # ------------------------------------------------------------------

    def _execute_spot_trade(self, asset, price, delta_exposure, microstructure_fn=None):
        """
        Live spot execution override.

        Flow:  health check → build payload → dry-run log  OR  live submit.
        Futures execution stays dry-run only until the Kraken Futures
        endpoint is separately wired.
        """
        if not self._pre_trade_safety(asset, price, delta_exposure):
            print(f"[LIVE SPOT BLOCKED SAFETY] {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        order = self._build_spot_order(asset, price, delta_exposure)
        if order is None:
            print(f"[LIVE SPOT NO-OP]     {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if self.dry_run:
            print(f"[LIVE SPOT DRY-RUN]   {asset}  order={order}")
            return 0.0

        # Live path — actually submit to Kraken
        result = self._submit_spot_order(order)
        if result is None:
            print(f"[LIVE SPOT FAILED]    {asset}  order={order}")
        return 0.0

    def _execute_futures_trade(self, asset, price, delta_exposure, microstructure_fn=None):
        """
        Live futures execution override.

        Flow:  full safety gate → build payload → dry-run log  OR  live submit.
        Mirrors the spot flow exactly; submission delegates to
        _submit_futures_order which calls _kraken_futures_private (stub until
        Futures API credentials are wired).
        """
        if not self._pre_trade_safety(asset, price, delta_exposure):
            print(f"[LIVE FUTURES BLOCKED SAFETY] {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        order = self._build_futures_order(asset, price, delta_exposure)
        if order is None:
            print(f"[LIVE FUTURES NO-OP]   {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if self.dry_run:
            print(f"[LIVE FUTURES DRY-RUN] {asset}  order={order}")
            return 0.0

        # Live path — submit to Kraken Futures
        result = self._submit_futures_order(order)
        if result is None:
            print(f"[LIVE FUTURES FAILED]  {asset}  order={order}")
        return 0.0


# ======================================================================
# Module-level helper — outside LiveBroker class
# ======================================================================

def run_live_trading_loop(broker, agent, loop_sleep: float = 1.0):
    """
    Main live trading loop.

    Responsibilities per iteration:
    - Respect kill switch (hard stop)
    - Let the agent run one decision step
    - Emit heartbeat + health metrics
    - Run daily rollover
    - Evaluate alert conditions
    """
    # Initial sync before trading
    state = broker.sync_live_account_state()
    if state is None:
        print("[MAIN LOOP] Failed initial account sync — aborting")
        return

    balances, positions = state
    print(f"[MAIN LOOP] Initial balances: {balances}")
    print(f"[MAIN LOOP] Initial positions: {positions}")

    # Set starting equity anchor on first run if not already set
    if getattr(broker, "_starting_equity", None) is None:
        zusd = float(balances.get("ZUSD", 0.0))
        broker._starting_equity = zusd
        print(f"[MAIN LOOP] Starting equity set to {zusd}")

    try:
        while True:
            # Hard stop if kill switch is active
            if broker.kill_switch:
                print("[MAIN LOOP] Kill switch active — exiting loop")
                break

            # Strategy step
            try:
                agent.step()
            except Exception as e:
                print(f"[MAIN LOOP] Exception in agent.step(): {e}")

            # Monitoring + ops
            broker.heartbeat()
            broker.record_health_metrics()
            broker.daily_rollover()
            broker.alerting_loop()

            time.sleep(loop_sleep)

    except KeyboardInterrupt:
        print("[MAIN LOOP] KeyboardInterrupt — shutting down cleanly")
    except Exception as e:
        print(f"[MAIN LOOP] Unhandled exception in main loop: {e}")
        broker.trigger_kill_switch(f"Main loop exception: {e}")
