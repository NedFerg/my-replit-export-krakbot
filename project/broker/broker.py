import os
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Kraken API credentials from the environment — never hard-coded.
        self.kraken_api_key    = os.getenv("KRAKEN_API_KEY", "")
        self.kraken_api_secret = os.getenv("KRAKEN_API_SECRET", "")

        if not self.kraken_api_key or not self.kraken_api_secret:
            print("[LiveBroker] WARNING: KRAKEN_API_KEY / KRAKEN_API_SECRET not set. "
                  "Running in dry-run mode with no credentials.")

        # Safety: always start in dry-run — flip only after full wiring + checks.
        self.dry_run = True

        # Health and kill-switch state
        self.last_api_error   = None   # last exception or error message from API
        self.last_latency_sec = 0.0    # round-trip latency of last API call
        self.kill_switch      = False  # set True to halt all order flow immediately

        # Kraken REST configuration
        self.kraken_base_url = "https://api.kraken.com"
        self.kraken_session  = requests.Session()

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
    # Live order execution (overrides SimulatedBroker private helpers)
    # Signature must match the parent: (asset, price, delta_exposure, microstructure_fn)
    # ------------------------------------------------------------------

    def _execute_spot_trade(self, asset, price, delta_exposure, microstructure_fn=None):
        """
        Live spot execution override.

        Gate order: health check → build payload → dry-run log.
        TODO: when dry_run is disabled, submit via
              _kraken_private("/0/private/AddOrder", order)
        """
        if not self.check_health():
            print(f"[LIVE SPOT BLOCKED]   {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        order = self._build_spot_order(asset, price, delta_exposure)
        if order is None:
            print(f"[LIVE SPOT NO-OP]     {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if self.dry_run:
            print(f"[LIVE SPOT DRY-RUN]   {asset}  order={order}")
            return 0.0

        # TODO: submit to Kraken spot endpoint
        print(f"[LIVE SPOT ORDER]     {asset}  order={order}")
        return 0.0

    def _execute_futures_trade(self, asset, price, delta_exposure, microstructure_fn=None):
        """
        Live futures execution override.

        Gate order: health check → build payload → dry-run log.
        TODO: when dry_run is disabled, submit to the Kraken Futures endpoint.
        """
        if not self.check_health():
            print(f"[LIVE FUTURES BLOCKED] {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        order = self._build_futures_order(asset, price, delta_exposure)
        if order is None:
            print(f"[LIVE FUTURES NO-OP]   {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if self.dry_run:
            print(f"[LIVE FUTURES DRY-RUN] {asset}  order={order}")
            return 0.0

        # TODO: submit to Kraken Futures/perps endpoint
        print(f"[LIVE FUTURES ORDER]   {asset}  order={order}")
        return 0.0
