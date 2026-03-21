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
from utils.market_hours import MarketHours, MarketSession
from broker.etf_hedging import ETFHedger, ETF_ASSETS

from utils.constants import (
    is_etf_tradeable,
    get_etf_order_type,
    MarketPeriod,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
)
from broker.etf_hedging import (
    ETFHedgingLayer,
    ETFOrder,
    ALL_ETFS,
    ETF_KRAKEN_PAIRS,
    ETFMode,
)


# ===========================================================================
# ASSET ALLOWLIST — only these tickers may ever be traded by this bot.
#
# Even if the Kraken API key has "Allow trading of stocks and ETFs" enabled
# in the account security settings, the bot will NEVER submit an order for
# a stock or non-crypto instrument.  Any asset name not in this set is
# rejected before reaching the order-builder or Kraken API, with a clear
# log message explaining the rejection.
#
# To add a new asset: extend this set AND add it to self.kraken_pairs in
# LiveBroker.__init__ so price feeds and order routing also know about it.
# DO NOT add stock tickers here — this bot is crypto-only.
# ===========================================================================
APPROVED_ASSETS: frozenset[str] = frozenset({
    # --- Spot crypto (the universe the bot watches and trades) -----------
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LINK",
    "HBAR",
    "XLM",
    "AVAX",
    # --- Kraken crypto ETP / leveraged tokens ----------------------------
    # These are crypto-underlying products listed on Kraken's ETP platform.
    # They are NOT stocks — they are crypto-collateralised structured tokens.
    "ETHU",   # ETH 2× Long ETP
    "ETHD",   # ETH 2× Short ETP
    "SLON",   # SOL 2× Long ETP
    "XXRP",   # XRP 2× Long ETP
    "SETH",   # ETH 1× Short ETP
})


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
    Live trading broker for Kraken spot.

    Inherits the full hybrid execution engine from SimulatedBroker so that
    strategy logic (vol targeting, leverage limits, hedging, slippage) is
    shared between sim and live modes without duplication.

    Safety defaults
    ---------------
    - KRAKEN_SANDBOX=true  (default) → connects to api.kraken.com but sends
      validate=true on all order submissions so Kraken validates the payload
      without executing it.  Read operations (prices, balances) hit the real
      API so data is accurate.
      Set KRAKEN_SANDBOX=false in Replit Secrets only when ready for live money.
    - ENABLE_FUTURES=False (class constant) → all futures code paths are fully
      disabled.  US residents cannot legally trade Kraken Futures.  Set to True
      only if you are in a permitted jurisdiction and have separate futures keys.
    - dry_run=True by default; set False only after confirming credentials.
    - kill-switch halts all order flow immediately.
    """

    ENABLE_FUTURES: bool = False

    # ETP pairs that are only available on Kraken's public Ticker endpoint
    # during US market hours (Mon–Fri, including pre/after-market sessions).
    # Outside those hours they return EQuery:Unknown asset pair.
    _ETP_ASSETS: frozenset = frozenset({"ETHU", "SLON", "XXRP", "ETHD", "SETH"})

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

        # Fee accounting
        # Kraken spot taker fee for retail tier (< $10k 30-day volume).
        # Maker = 0.16 %, taker = 0.40 %.  Limit orders earn maker rebate when
        # they rest on the book; we default to limit orders now.
        # Round-trip taker cost = 2 × 0.40 % = 0.80 % per trade.
        self.taker_fee           = 0.004    # 0.40 % taker (retail tier < $10k/30d)
        self.maker_fee           = 0.0016   # 0.16 % maker (earned on limit orders)
        self.cumulative_fees_usd = 0.0      # total estimated fees paid this session

        # --- Sandbox vs. Production ------------------------------------------
        # Default: SANDBOX.  Switch to live only by setting KRAKEN_SANDBOX=false
        # in Replit Secrets.  This is intentional — sandbox is the safe default.
        _use_sandbox = os.getenv("KRAKEN_SANDBOX", "true").strip().lower()
        self._sandbox_mode = _use_sandbox not in ("false", "0", "no")
        # Kraken has no separate sandbox domain.  In sandbox mode we still
        # connect to api.kraken.com for price/balance reads, but every order
        # submission carries validate=true so Kraken validates the payload
        # without executing it.
        self.kraken_base_url = "https://api.kraken.com"
        if self._sandbox_mode:
            print("[LiveBroker] SANDBOX MODE — api.kraken.com (validate=true on orders).")
            print("  To go live: set KRAKEN_SANDBOX=false in Replit Secrets.")
        else:
            print("[LiveBroker] LIVE MODE — connected to api.kraken.com")

        # Limit-order tolerance: how far from mid-price the limit is placed.
        # 0.1 % keeps fills fast on liquid pairs while protecting against
        # wide-spread/thin-book execution on HBAR and XLM.
        self.limit_order_tolerance = float(
            os.getenv("LIMIT_ORDER_TOLERANCE", "0.001")
        )

        self.kraken_session = requests.Session()

        # --- Futures configuration -------------------------------------------
        # ENABLE_FUTURES is a class-level constant (False by default).
        # US residents CANNOT legally trade Kraken Futures.
        # To enable: change the class constant to True AND add dedicated
        # KRAKEN_FUTURES_API_KEY / KRAKEN_FUTURES_API_SECRET secrets.
        # All futures attributes are still initialised to safe zero-state so
        # the rest of the class does not need None-guards everywhere.
        self.futures_available      = False   # hard gate: mirrors ENABLE_FUTURES
        self.futures_paper_mode     = True    # always paper when disabled
        self.futures_api_key        = ""
        self.futures_api_secret     = ""
        self.futures_base_url       = "https://futures.kraken.com/derivatives/api/v3"
        self.futures_wallet_usd     = 0.0
        self.futures_max_leverage   = 2.0
        self.last_futures_wallet_ts = 0.0

        if self.ENABLE_FUTURES:
            has_futures_keys = bool(os.getenv("KRAKEN_FUTURES_API_KEY"))
            self.futures_paper_mode = not has_futures_keys
            self.futures_api_key    = (os.getenv("KRAKEN_FUTURES_API_KEY") or "")
            self.futures_api_secret = (os.getenv("KRAKEN_FUTURES_API_SECRET") or "")
            self.futures_available  = True
            if self.futures_paper_mode:
                print("[FUTURES] PAPER mode — no live orders sent to Kraken.")
            else:
                print("[FUTURES] LIVE mode — dedicated futures keys found.")
        else:
            print("[FUTURES] DISABLED — ENABLE_FUTURES=False. "
                  "All futures code paths are fully suppressed.")

        # Map internal asset names → Kraken ticker symbols used for the
        # public /0/public/Ticker price-feed batch request.
        # Crypto spot pairs use Kraken's X-prefixed legacy names.
        # ETF/ETP tickers on Kraken use bare ticker symbols (no currency suffix).
        self.kraken_pairs = {
            "BTC":  "XBTUSD",
            "ETH":  "ETHUSD",
            "SOL":  "SOLUSD",
            "AVAX": "AVAXUSD",
            "LINK": "LINKUSD",
            "HBAR": "HBARUSD",
            "XRP":  "XRPUSD",
            "XLM":  "XLMUSD",
            # ETF hedging instruments (spot-traded, Kraken Spot) — bare ticker symbols
            # Long ETFs
            "ETHU": "ETHU",      # ETH 2× Long ETP
            "SLON": "SLON",      # SOL 2× Long ETP
            "XXRP": "XXRP",      # XRP 2× Long ETP
            # Short ETFs (only these two are short)
            "ETHD": "ETHD",      # ETH 2× Short ETP
            "SETH": "SETH",      # ETH 1× Short ETP
        }

        # Kraken Futures perpetual contract symbols (PF_ = linear / USD-settled).
        # Only assets with liquid Kraken Futures contracts are listed here;
        # assets absent from this map are skipped by run_futures_overlay().
        self.kraken_futures_pairs = {
            "BTC":  "PF_XBTUSD",
            "ETH":  "PF_ETHUSD",
            "SOL":  "PF_SOLUSD",
            "XRP":  "PF_XRPUSD",
            "LINK": "PF_LINKUSD",
            "AVAX": "PF_AVAXUSD",
            # HBAR and XLM do not have Kraken Futures perpetuals — skipped.
        }

        # Kraken balance-dict keys for each tracked asset
        # (Kraken prefixes some with X; LINK/HBAR/AVAX/SOL use bare names;
        # ETP tickers use their plain ticker as the balance key)
        self.kraken_balance_keys = {
            "BTC":  "XXBT",
            "ETH":  "XETH",
            "SOL":  "SOL",
            "AVAX": "AVAX",
            "LINK": "LINK",
            "HBAR": "HBAR",
            "XRP":  "XXRP",
            "XLM":  "XXLM",
            "ETHU": "ETHU",
            "SLON": "SLON",
            "XXRP": "XXRP",
            "ETHD": "ETHD",
            "SETH": "SETH",
        }

        # --- ETF hedging layer -------------------------------------------
        # ETF positions are tracked separately from crypto spot positions
        # so the 24/7 spot layer and the market-hours-aware ETF layer never
        # interfere with each other.
        self.etf_positions: dict = {a: 0.0 for a in ETF_ASSETS}   # asset → coin qty
        self.etf_hedger            = ETFHedger(
            max_etf_allocation=float(os.getenv("MAX_ETF_ALLOCATION", "0.30")),
        )
        self._market_hours         = MarketHours()

        # Kraken balance keys for ETF assets (bare ticker on Kraken Spot)
        self.kraken_balance_keys_etf = {
            "ETHD": "ETHD",
            "SETH": "SETH",
        }

        # Live price cache (populated by fetch_live_prices)
        self.live_prices           = {}   # asset → latest float price
        self.last_price_timestamp  = 0    # Unix timestamp of last successful fetch

        # Live account state (read-only in dry-run)
        self.live_balances  = {}   # raw Kraken balance dict
        self.live_positions = {}   # raw Kraken open-positions dict

        # --- ETF position tracking (separate from crypto spot positions) -----
        # etf_positions stores coin quantities for leveraged ETF holdings.
        # These are spot orders on Kraken but managed separately so the 30%
        # cap can be enforced independently of crypto spot positions.
        self.etf_positions: dict = {etf: 0.0 for etf in ALL_ETFS}

        # ETF hedging/amplification layer — handles regime logic, order
        # sizing, market-hours awareness, and the 30% portfolio cap.
        self.etf_layer = ETFHedgingLayer()

        # --- Automatic safety limits (first-night harness) ---------------
        self.max_notional_per_asset = 50.0    # USD cap per single asset trade
        self.max_total_notional     = 200.0   # USD cap across all open positions
        self.max_trades_per_hour    = 50      # 5-min cooldown × 4 orders/round = 48 max/hr

        # Daily loss cap expressed as a fraction of starting equity.
        # 10 % of equity is a sensible hard stop for an algorithmic bot.
        # The absolute USD cap is computed dynamically once starting equity is
        # known (see _check_daily_loss).  This avoids the "fixed $50 cap on a
        # $136 account = 37 % drawdown" bug from the previous hard-coded value.
        self.max_daily_loss_pct  = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.10"))
        self._max_daily_loss_usd = None   # set dynamically from starting equity

        # Rate-gate intervals for monitoring calls (seconds).
        # Prevents record_health_metrics() and alerting_loop() from printing
        # verbose output and running equity calculations every 1-second loop tick.
        self._health_metrics_interval = int(os.getenv("HEALTH_METRICS_INTERVAL", "60"))
        self._alerting_interval       = int(os.getenv("ALERTING_INTERVAL",       "60"))
        self._last_health_metrics_ts  = 0.0
        self._last_alerting_ts        = 0.0

        self._trade_count_window = []   # timestamps of recent trades (rolling 1 h)
        self._starting_equity    = None # total equity at first live trade of the session

    # ------------------------------------------------------------------
    # Fee tier helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_kraken_taker_fee(trading_volume_30d: float) -> float:
        """
        Return the Kraken spot taker fee rate for the given 30-day USD volume.

        Tier schedule (as of 2025):
            < $10,000  → 0.40 %
            < $50,000  → 0.35 %
            < $100,000 → 0.24 %
            < $250,000 → 0.22 %
            < $500,000 → 0.20 %
            < $1M      → 0.18 %
            < $2.5M    → 0.16 %
            < $5M      → 0.14 %
            ≥ $5M      → 0.12 %

        Parameters
        ----------
        trading_volume_30d : Total USD notional traded in the last 30 days.

        Returns
        -------
        float — taker fee as a decimal fraction (e.g. 0.004 for 0.40 %).
        """
        if trading_volume_30d < 10_000:
            return 0.0040
        elif trading_volume_30d < 50_000:
            return 0.0035
        elif trading_volume_30d < 100_000:
            return 0.0024
        elif trading_volume_30d < 250_000:
            return 0.0022
        elif trading_volume_30d < 500_000:
            return 0.0020
        elif trading_volume_30d < 1_000_000:
            return 0.0018
        elif trading_volume_30d < 2_500_000:
            return 0.0016
        elif trading_volume_30d < 5_000_000:
            return 0.0014
        else:
            return 0.0012

    def sync_fee_tier_from_kraken(self) -> float:
        """
        Fetch the user's current 30-day trading volume directly from Kraken and
        update self.taker_fee / self.maker_fee to match the live tier.

        Calls the private /0/private/TradeVolume endpoint, which returns the
        authenticated user's 30-day USD volume.  Falls back to the current
        self.taker_fee value on any error (credentials missing, network issue).

        Returns
        -------
        float — the taker fee rate now in effect (e.g. 0.004 for 0.40 %).
        """
        result = self._kraken_private("/0/private/TradeVolume", {"fee-info": "true"})
        if not result or "result" not in result:
            print("[LiveBroker] sync_fee_tier_from_kraken: API call failed — keeping current tier")
            return self.taker_fee

        volume_usd = 0.0
        try:
            volume_usd = float(result["result"].get("volume", 0.0))
        except (KeyError, ValueError, TypeError):
            print("[LiveBroker] sync_fee_tier_from_kraken: unexpected response format")
            return self.taker_fee

        new_taker = self.get_kraken_taker_fee(volume_usd)
        # Maker fee is typically 60 % of taker for Kraken spot (0.40 % → 0.16 %)
        new_maker = round(new_taker * 0.40, 6)

        if new_taker != self.taker_fee:
            print(
                f"[LiveBroker] Fee tier updated: "
                f"taker {self.taker_fee*100:.3f}% → {new_taker*100:.3f}%  "
                f"(30d volume: ${volume_usd:,.0f})"
            )

        self.taker_fee = new_taker
        self.maker_fee = new_maker
        return self.taker_fee

    # ------------------------------------------------------------------
    # Portfolio exposure — live override
    # ------------------------------------------------------------------

    def _compute_portfolio_exposure(self):
        """
        Override for LiveBroker: spot_positions stores raw unit quantities
        (e.g. 300 XLM, 0.5 SOL) rather than fractional exposures, so we
        must convert using live prices and total equity before summing.
        Futures positions are already stored as fractions (unchanged).
        """
        equity = self.compute_total_equity() if hasattr(self, "compute_total_equity") else 0.0
        if equity <= 0:
            return 0.0, 0.0, 0.0

        total_spot = 0.0
        for asset, qty in self.spot_positions.items():
            price = self.live_prices.get(asset, 0.0)
            if price > 0:
                total_spot += abs(qty * price / equity)

        total_fut = sum(abs(v) for v in self.futures_positions.values())
        return total_spot, total_fut, total_spot + total_fut

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

    def validate_credentials(self) -> bool:
        """
        Confirm that KRAKEN_API_KEY and KRAKEN_API_SECRET are valid by calling
        the authenticated ``/0/private/Balance`` endpoint.

        Returns True on success (credentials accepted by Kraken) or False if
        the credentials are missing, malformed, or rejected.  On failure the
        kill switch is **not** triggered here — the caller decides whether to
        abort or fall back to paper mode.

        Intended to be called once at startup from ``go_live.sh`` / main.py
        before enabling live order submission.
        """
        if not self.kraken_api_key or not self.kraken_api_secret:
            print("[LiveBroker] validate_credentials: no API credentials configured.")
            return False

        print("[LiveBroker] Validating API credentials against Kraken…")
        result = self._kraken_private("/0/private/Balance")
        if result is None:
            print("[LiveBroker] validate_credentials: request failed (network error).")
            return False

        errors = result.get("error", [])
        if errors:
            print(f"[LiveBroker] validate_credentials: Kraken rejected credentials — {errors}")
            return False

        # Success — log the USD balance so the operator can confirm the right account
        usd_balance = float(result.get("result", {}).get("ZUSD", 0.0))
        print(
            f"[LiveBroker] ✅  Credentials valid — Kraken account balance: "
            f"${usd_balance:,.2f} USD"
        )
        return True

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
        Fetch the latest last-trade price for all tracked assets via Kraken's
        public Ticker endpoint in a SINGLE batched HTTP request.

        'c' field = last trade closed: [price, lot_volume].
        Returns the result dict and updates self.live_prices on success.
        Logs all fetched prices and warns loudly if any deviate from
        PRICE_SANITY_RANGES (configurable, defaults to known good ranges).
        """
        # Known-good reference ranges:  (min, max) USD per asset.
        # Override any entry via env vars: e.g. BTC_PRICE_MIN=60000
        # These are generous ±50 % bounds around typical market prices so the
        # check flags impossible values (wrong pair, wrong currency, stale cache)
        # without triggering on normal market moves.
        PRICE_SANITY_RANGES: dict[str, tuple[float, float]] = {
            "BTC":  (float(os.getenv("BTC_PRICE_MIN",  "30000")),
                     float(os.getenv("BTC_PRICE_MAX",  "200000"))),
            "ETH":  (float(os.getenv("ETH_PRICE_MIN",  "500")),
                     float(os.getenv("ETH_PRICE_MAX",  "10000"))),
            "SOL":  (float(os.getenv("SOL_PRICE_MIN",  "10")),
                     float(os.getenv("SOL_PRICE_MAX",  "1000"))),
            "XRP":  (float(os.getenv("XRP_PRICE_MIN",  "0.10")),
                     float(os.getenv("XRP_PRICE_MAX",  "20"))),
            "LINK": (float(os.getenv("LINK_PRICE_MIN", "1")),
                     float(os.getenv("LINK_PRICE_MAX", "100"))),
            "HBAR": (float(os.getenv("HBAR_PRICE_MIN", "0.01")),
                     float(os.getenv("HBAR_PRICE_MAX", "1"))),
            "XLM":  (float(os.getenv("XLM_PRICE_MIN",  "0.01")),
                     float(os.getenv("XLM_PRICE_MAX",  "5"))),
        }

        # ETP pairs (ETHU, SLON, XXRP, ETHD, SETH) are only listed on Kraken's
        # public Ticker endpoint during US market hours (Mon–Fri 09:30–16:30 ET,
        # plus pre/after-market).  Requesting them outside those hours returns
        # EQuery:Unknown asset pair.  Core crypto pairs are available 24/7.
        if self._market_hours.etf_trading_allowed():
            active_pairs = self.kraken_pairs
        else:
            active_pairs = {k: v for k, v in self.kraken_pairs.items()
                            if k not in self._ETP_ASSETS}

        # Build comma-separated pair string once
        pairs_str = ",".join(active_pairs.values())
        data = self._kraken_public("/0/public/Ticker", {"pair": pairs_str})

        if not data or "result" not in data:
            print("[PRICE FEED ERROR] Batch ticker request failed or returned no result")
            return {}

        ticker_data = data["result"]
        result      = {}

        # Build a reverse map: Kraken-normalised-pair → internal asset name.
        # Kraken normalises many pairs on the way out (e.g. XBTUSD → XXBTZUSD,
        # ETHUSD → XETHZUSD, XRPUSD → XXRPZUSD, XLMUSD → XXLMZUSD).
        _KRAKEN_ALIASES: dict = {
            "XXBTZUSD": "BTC",   # Kraken normalises XBTUSD  → XXBTZUSD
            "XETHZUSD": "ETH",   # Kraken normalises ETHUSD  → XETHZUSD
            "XXRPZUSD": "XRP",   # Kraken normalises XRPUSD  → XXRPZUSD
            "XXLMZUSD": "XLM",   # Kraken normalises XLMUSD  → XXLMZUSD
        }
        pair_to_asset = {v.upper(): k for k, v in self.kraken_pairs.items()}
        pair_to_asset.update(_KRAKEN_ALIASES)   # aliases override if there is a clash

        for kraken_key, ticker in ticker_data.items():
            asset = pair_to_asset.get(kraken_key.upper())
            if asset is None:
                print(f"[PRICE FEED] Unknown pair key from Kraken: {kraken_key!r} — "
                      f"add to _KRAKEN_ALIASES if needed")
                continue
            try:
                result[asset] = float(ticker["c"][0])
            except Exception:
                print(f"[PRICE PARSE ERROR] Could not parse last trade for {kraken_key}")

        if result:
            self.live_prices          = result
            self.last_price_timestamp = time.time()

            # ---- Log every fetched price so the feed is fully auditable ----
            price_line = "  ".join(
                f"{a}=${result[a]:,.2f}"
                for a in ["BTC", "ETH", "SOL", "XRP", "HBAR", "LINK", "XLM",
                           "ETHD", "SETH"]
                if a in result
            )
            print(f"[PRICE FEED] {price_line}")

            # ---- Sanity check — flag prices outside known-good ranges ------
            bad = []
            for asset, (lo, hi) in PRICE_SANITY_RANGES.items():
                price = result.get(asset)
                if price is not None and not (lo <= price <= hi):
                    bad.append(f"{asset}=${price:,.2f} (expected ${lo:,.0f}–${hi:,.0f})")
            if bad:
                print(
                    f"[PRICE FEED ⚠️  SANITY FAIL] The following prices are outside "
                    f"expected ranges — verify your Kraken connection is returning "
                    f"real spot USD prices:\n  " + "\n  ".join(bad)
                )

            missing = [a for a in active_pairs if a not in result]
            if missing:
                print(f"[PRICE FEED] Missing prices for: {missing}")

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
        on failure.  Retries once with back-off on rate-limit errors.
        """
        for attempt in range(2):
            data = self._kraken_private("/0/private/Balance")
            if not data:
                break
            errors = data.get("error", [])
            if any("Rate limit" in e for e in errors):
                print(f"[BALANCE] Rate limit hit — waiting 15 s before retry (attempt {attempt+1})")
                time.sleep(15)
                continue
            if "result" not in data:
                break
            self.live_balances = data["result"]
            return self.live_balances
        print("[BALANCE ERROR] Could not fetch balances from Kraken")
        return None

    def fetch_live_positions(self):
        """
        Fetch open positions (spot-margin or futures) from Kraken (read-only).
        Populates self.live_positions and returns the raw result dict, or None
        on failure.  Retries once with back-off on rate-limit errors.
        """
        for attempt in range(2):
            data = self._kraken_private("/0/private/OpenPositions", {"docalcs": "true"})
            if not data:
                break
            errors = data.get("error", [])
            if any("Rate limit" in e for e in errors):
                print(f"[POSITION] Rate limit hit — waiting 15 s before retry (attempt {attempt+1})")
                time.sleep(15)
                continue
            if "result" not in data:
                break
            self.live_positions = data["result"]
            return self.live_positions
        print("[POSITION ERROR] Could not fetch open positions from Kraken")
        return None

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

        Paper mode: no HTTP call — paper positions are maintained entirely by
        run_futures_overlay(); returns the current dict unchanged.
        Live mode: calls _kraken_futures_private → /openpositions.
        Populates self.futures_positions with {asset: fractional_delta}.
        """
        if self.futures_paper_mode:
            print(f"[PAPER FUTURES POSITIONS] {self.futures_positions}")
            return self.futures_positions

        resp = self._kraken_futures_private("/openpositions", {}, method="GET")

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

        # Futures exposure.
        # futures_positions stores fractional deltas (fraction of equity, signed).
        # Convert to USD by multiplying by total equity so leverage reads correctly.
        fut_equity = self.compute_total_equity()
        for asset, frac in (self.futures_positions or {}).items():
            try:
                futures_exposure[asset] = float(frac) * fut_equity
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
        equity = self.compute_total_equity()

        exposure       = self.compute_unified_exposure()
        total_notional = exposure.get("total_notional", 0.0)

        starting    = self._starting_equity
        session_pnl = None
        if starting is not None:
            session_pnl = equity - float(starting)

        leverage = None
        if equity > 0:
            leverage = total_notional / equity

        fees_paid = getattr(self, "cumulative_fees_usd", 0.0)
        fee_adj_pnl = (session_pnl - fees_paid) if session_pnl is not None else None

        snapshot = {
            "equity":             equity,
            "starting_equity":    starting,
            "session_pnl":        session_pnl,
            "fees_paid":          round(fees_paid, 4),
            "fee_adjusted_pnl":   round(fee_adj_pnl, 4) if fee_adj_pnl is not None else None,
            "total_notional":     total_notional,
            "net_exposure":       exposure.get("net_exposure", 0.0),
            "leverage":           leverage,
            "spot_exposure":      exposure.get("spot_exposure", {}),
            "futures_exposure":   exposure.get("futures_exposure", {}),
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
            "fees_paid":        pnl.get("fees_paid"),
            "fee_adj_pnl":      pnl.get("fee_adjusted_pnl"),
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
                print(f"[HEARTBEAT] alive at {time.strftime('%Y-%m-%d %H:%M:%S')} local  |  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
                self.emit_health_check()
            self._last_heartbeat = now

    def record_health_metrics(self):
        """
        Record a single snapshot of key risk metrics.

        Rate-gated: fires at most once per _health_metrics_interval seconds
        (default 60 s).  This prevents the 1-second main loop from filling the
        health log in 33 minutes and spamming the console with equity printouts.
        At 60-second cadence the 2000-entry log covers ~33 hours of history.

        Called automatically by heartbeat() or manually for forced snapshots.
        """
        now = time.time()
        if now - self._last_health_metrics_ts < self._health_metrics_interval:
            return
        self._last_health_metrics_ts = now

        if not hasattr(self, "_health_log"):
            self._health_log = []

        snapshot = self.emit_health_check()
        snapshot["timestamp"] = now

        # Keep last 2000 entries (~33 h at 60 s cadence)
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
                     heartbeat_timeout:  float = 600.0):
        """
        Evaluate alert conditions and print alerts when thresholds are crossed.

        Drawdown and exposure thresholds are derived from live account equity so
        they scale correctly regardless of account size.  No more hard-coded
        $500 drawdown limit that would never fire on a $136 account.
        """
        # --- 1. Kill switch alert ---
        if self.kill_switch:
            print("[ALERT] Kill switch active — trading halted")

        pnl = self.compute_unified_pnl_snapshot()

        # --- 2. Leverage alert ---
        lev = pnl.get("leverage")
        if lev is not None and lev > max_leverage:
            print(f"[ALERT] Leverage {lev:.2f} exceeds limit {max_leverage:.1f}×")

        # --- 3. Drawdown alert — uses the same equity-scaled cap as _check_daily_loss ---
        session_pnl   = pnl.get("session_pnl")
        equity        = pnl.get("equity") or 0.0
        # Alert at 50 % of the kill-switch cap so there is advance warning
        alert_cap_usd = (self._max_daily_loss_usd or (equity * self.max_daily_loss_pct)) * 0.5
        if session_pnl is not None and alert_cap_usd > 0 and session_pnl < -alert_cap_usd:
            print(f"[ALERT] Drawdown ${session_pnl:.2f} — approaching daily loss cap "
                  f"(alert at ${alert_cap_usd:.2f}, kill at "
                  f"${(self._max_daily_loss_usd or alert_cap_usd * 2):.2f})")

        # --- 4. Exposure alert — 150 % of account equity is a sensible ceiling ---
        net_exp    = pnl.get("net_exposure")
        exp_limit  = equity * 1.5 if equity > 0 else self.max_total_notional
        if net_exp is not None and abs(net_exp) > exp_limit:
            print(f"[ALERT] Net exposure ${net_exp:.2f} exceeds {exp_limit:.2f} (150% equity)")

        # --- 5. Heartbeat silence alert ---
        now = time.time()
        if hasattr(self, "_last_heartbeat"):
            if now - self._last_heartbeat > heartbeat_timeout:
                print("[ALERT] Heartbeat silence — engine may be frozen")

    def alerting_loop(self):
        """
        Rate-gated wrapper — fires at most once per _alerting_interval seconds.
        Prevents check_alerts() from running expensive equity calculations every
        1-second loop tick.
        """
        now = time.time()
        if now - self._last_alerting_ts < self._alerting_interval:
            return
        self._last_alerting_ts = now
        self.check_alerts()

    # ------------------------------------------------------------------
    # Order payload builders (formatting only — nothing is submitted)
    # ------------------------------------------------------------------

    @staticmethod
    def _fractional_to_coin_units(delta_frac: float, equity_usd: float,
                                  price: float) -> float:
        """
        Convert a fractional exposure delta to base-asset coin units.

        delta_frac  : signed fraction of total equity  (e.g. +0.06 = buy 6% of equity)
        equity_usd  : total portfolio value in USD
        price       : current asset price in USD/coin

        Returns the unsigned coin quantity; sign is communicated separately
        as the order side ("buy" / "sell").

        This function is the single authoritative conversion point — every code
        path that builds a Kraken order payload MUST go through here so there
        is no ambiguity about what `volume` means.
        """
        notional_usd = abs(delta_frac) * equity_usd
        if price <= 0:
            return 0.0
        return notional_usd / price

    def _build_spot_order(self, asset: str, price: float,
                          coin_units: float, side: str) -> dict | None:
        """
        Format a Kraken spot order payload.

        For regular crypto assets, a LIMIT order is always used.
        For ETF assets (ETHD, SETH), the order type is determined by
        MarketHours: market orders during regular hours, limit orders
        during pre/after-market sessions.

        Parameters
        ----------
        asset      : internal asset name (e.g. "SOL", "ETHD")
        price      : current mid-price in USD
        coin_units : unsigned quantity in BASE-ASSET UNITS (e.g. 0.5 SOL).
                     Must already be converted from fractional exposure via
                     _fractional_to_coin_units().  Never pass a raw fraction here.
        side       : "buy" or "sell"

        Limit price
        -----------
        Buys  → mid × (1 + tolerance)  : we pay up to tolerance% above mid.
        Sells → mid × (1 - tolerance)  : we accept down to tolerance% below mid.
        Default tolerance = 0.1% (self.limit_order_tolerance).

        This replaces unconditional market orders.  On liquid pairs the limit
        fills immediately at or better than the limit price; on thin books it
        prevents catastrophic slippage on HBAR/XLM.
        Limit orders also qualify for maker fee (0.16%) rather than taker (0.40%)
        when they rest on the book.

        Returns None for invalid inputs (triggers kill switch on unknown asset or
        bad price, so the caller does not need to check).
        """
        if asset not in self.kraken_pairs:
            self.trigger_kill_switch(f"Unknown asset for spot order: {asset}")
            return None

        if price is None or price <= 0:
            self.trigger_kill_switch(
                f"Invalid price for spot order: {asset} price={price}"
            )
            return None

        if coin_units <= 0:
            return None

        if side not in ("buy", "sell"):
            self.trigger_kill_switch(f"Invalid side for spot order: {side!r}")
            return None

        is_etf = asset in ETF_ASSETS
        if is_etf:
            # ETF assets: market hours determine the order type
            order_type = self._market_hours.required_order_type()
        else:
            # Crypto spot: always limit orders (maker rebate + slippage protection)
            order_type = "limit"

        tol = self.limit_order_tolerance
        if order_type == "limit":
            if side == "buy":
                limit_price = round(price * (1.0 + tol), 8)
            else:
                limit_price = round(price * (1.0 - tol), 8)
            return {
                "pair":      self.kraken_pairs[asset],
                "type":      side,
                "ordertype": "limit",
                "price":     f"{limit_price:.8f}",
                "volume":    f"{coin_units:.8f}",
            }
        else:
            # Market order (ETF, regular session only)
            return {
                "pair":      self.kraken_pairs[asset],
                "type":      side,
                "ordertype": "market",
                "volume":    f"{coin_units:.8f}",
            }

    def _build_futures_order(self, asset, price, delta_exposure):
        """
        Format a Kraken Futures perpetual order payload from a fractional delta.

        Uses kraken_futures_pairs (PF_ symbols) for the contract ticker.
        Assets not listed in that map (e.g. HBAR, XLM) are soft-skipped with
        None — no kill switch — because Kraken does not list them as futures.
        Returns None for unknown assets, zero-delta or invalid inputs.
        """
        futures_symbol = self.kraken_futures_pairs.get(asset)
        if futures_symbol is None:
            return None   # Asset has no futures contract (e.g. HBAR, XLM) — skip silently

        if price is None or price <= 0:
            self.trigger_kill_switch(f"Invalid price for futures order: {asset} price={price}")
            return None

        if delta_exposure == 0:
            return None

        # Convert fractional delta → coin units.
        # Sizing basis = futures wallet collateral × allowed leverage.
        # If the futures wallet hasn't been fetched yet, fall back to spot
        # equity so the overlay still produces a non-zero order.
        if self.futures_wallet_usd > 0:
            sizing_usd = self.futures_wallet_usd * self.futures_max_leverage
        else:
            sizing_usd = self.compute_total_equity() or 1.0

        # Kraken Futures size is in base-currency units (e.g. SOL for PF_SOLUSD).
        coin_size = round(abs(delta_exposure * sizing_usd) / price, 8)

        if coin_size == 0:
            return None

        return {
            "symbol":    futures_symbol,
            "side":      "buy" if delta_exposure > 0 else "sell",
            "orderType": "mkt",
            "size":      coin_size,
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

    def compute_total_equity(self) -> float:
        """
        True portfolio value = ZUSD cash + all crypto holdings + ETF holdings
        at live prices.  Avoids treating buy orders as losses in the daily-loss
        check.  ETF positions are tracked in ``self.etf_positions`` (coin units)
        and included here so daily loss caps scale correctly.
        """
        balances = self.live_balances or {}
        equity   = float(balances.get("ZUSD", 0.0))
        for asset, bal_key in self.kraken_balance_keys.items():
            qty   = float(balances.get(bal_key, 0.0))
            price = self.live_prices.get(asset, 0.0)
            equity += qty * price
        # Add ETF notional (separate dict, not in kraken_balance_keys)
        for etf, qty in self.etf_positions.items():
            price = self.live_prices.get(etf, 0.0)
            equity += abs(qty) * price   # abs: short-ETFs are held long on Kraken
        return equity

    def _check_daily_loss(self) -> bool:
        """
        Compare current total portfolio value to the session-start value.
        Uses compute_total_equity() so that buying crypto is not counted as a loss.

        The loss cap is computed as (max_daily_loss_pct × starting_equity) the
        first time starting_equity is set, giving a threshold that scales with
        actual account size rather than a fixed dollar figure.

        Triggers kill switch if the drawdown exceeds the computed cap.
        """
        equity = self.compute_total_equity()

        if self._starting_equity is None:
            if equity > 0:
                self._starting_equity    = equity
                self._max_daily_loss_usd = round(equity * self.max_daily_loss_pct, 2)
                print(f"[LOSS CAP] Starting equity anchored at ${equity:.2f}  "
                      f"→ daily loss cap = ${self._max_daily_loss_usd:.2f} "
                      f"({self.max_daily_loss_pct:.0%})")
            return True

        if equity <= 0:
            return True   # no balance info yet — permissive

        loss_cap = self._max_daily_loss_usd or (self._starting_equity * self.max_daily_loss_pct)
        if equity < self._starting_equity - loss_cap:
            self.trigger_kill_switch(
                f"Daily loss limit exceeded: equity={equity:.2f}  "
                f"start={self._starting_equity:.2f}  cap=${loss_cap:.2f} "
                f"({self.max_daily_loss_pct:.0%})"
            )
            return False

        return True

    def _pre_trade_safety(self, asset: str, price: float,
                          notional_usd: float) -> bool:
        """
        Unified pre-trade gate — runs all safety checks in priority order.
        Returns False (and triggers the kill switch where appropriate) if any
        check fails.  Called at the top of every live execution path.

        Checks (in order)
        -----------------
        1. Kill-switch / health               — halt if system is unhealthy
        2. Asset allowlist (APPROVED_ASSETS)  — reject non-crypto immediately
        3. Trade rate limiter                 — enforce order-per-hour cap
        4. Per-asset notional cap             — skip oversized single orders
        5. Total portfolio notional cap       — skip when book is full
        6. Daily loss cap                     — halt if drawdown limit hit

        Parameters
        ----------
        asset        : internal asset name ("SOL", "BTC", …)
        price        : current mid-price in USD (used only for logging)
        notional_usd : UNSIGNED USD notional of the proposed order.
                       Both callers must pass the same unit:
                         execute_trade      → size_coins × price
                         _execute_spot_trade → abs(delta_frac) × equity
        """
        if not self.check_health():
            return False

        # ----------------------------------------------------------------
        # ASSET ALLOWLIST — block anything not in the approved crypto set.
        # This fires before every other check so a rejected asset never
        # reaches the order-builder or Kraken API, regardless of what
        # permissions are enabled on the API key (including stock trading).
        # ----------------------------------------------------------------
        if asset not in APPROVED_ASSETS:
            print(
                f"[SAFETY] Order BLOCKED — '{asset}' is not in the approved "
                f"crypto asset list.  This bot trades only: "
                f"{sorted(APPROVED_ASSETS)}.  "
                f"No stock or non-crypto instrument will ever be submitted, "
                f"even if the API key has stock-trading permission enabled."
            )
            return False

        if not self._check_trade_rate():
            return False

        # Per-asset notional cap — soft reject (skip this order, don't halt the bot)
        # 1.001× threshold absorbs floating-point boundary rounding
        abs_notional = abs(notional_usd)
        if abs_notional > self.max_notional_per_asset * 1.001:
            print(f"[SAFETY] Per-asset cap: skipping {asset} "
                  f"(${abs_notional:.2f} > ${self.max_notional_per_asset:.2f}  "
                  f"price=${price:.4f})")
            return False

        # Total portfolio notional cap — soft reject
        if self._current_notional() > self.max_total_notional:
            print(f"[SAFETY] Total notional cap: skipping order "
                  f"(${self._current_notional():.2f} > ${self.max_total_notional:.2f})")
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
        size   : float Position size in BASE-ASSET COIN UNITS (positive).
                       The agent computes this via (notional_usd / price)
                       before calling place_order().

        Flow
        ----
        1. Dry-run guard — logs and returns immediately if dry_run=True
        2. Price validation — kill switch on missing / zero price
        3. Pre-trade safety harness — notional cap, rate limit, daily loss
        4. Build spot limit-order payload (coin units, correct side)
        5. Submit to Kraken /AddOrder
        6. Update internal position tracker and cumulative fee on success

        Returns the Kraken result dict on success, None on any failure.
        """
        if self.dry_run:
            notional = size * (self.live_prices.get(symbol) or 0.0)
            print(f"[DRY RUN] {side.upper()} {size:.6f} {symbol}"
                  f"  notional≈${notional:.2f}")
            return None

        price = self.live_prices.get(symbol)
        if not price or price <= 0:
            self.trigger_kill_switch(
                f"No valid live price for {symbol} — order aborted"
            )
            return None

        notional = size * price   # USD notional of this order (unsigned)

        if not self._pre_trade_safety(symbol, price, notional):
            return None

        # size is already in coin units — pass directly to the builder
        order = self._build_spot_order(symbol, price, size, side)
        if order is None:
            return None

        result = self._submit_spot_order(order)
        if result is not None:
            # Update position tracker (coin units, signed)
            signed_units = size if side == "buy" else -size
            self.spot_positions[symbol] = (
                self.spot_positions.get(symbol, 0.0) + signed_units
            )
            # Estimate fee (taker rate; may be lower if order rests as maker)
            fee_usd = notional * self.taker_fee
            self.cumulative_fees_usd += fee_usd
            print(f"[EXECUTE] {side.upper()} {size:.6f} {symbol} @ {price:.4f}"
                  f"  notional=${notional:.2f}  fee≈${fee_usd:.4f}"
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

        Sandbox mode: injects validate=true so Kraken validates the payload
        server-side without creating a real order.  This confirms the order
        structure, API credentials, pair name, and volume minimums are all
        correct — without spending money.
        """
        if not order:
            return None

        # Inject Kraken's validate flag in sandbox mode so the order is
        # checked but never executed.  Remove this field on the live path.
        if self._sandbox_mode:
            order = dict(order)      # shallow copy — don't mutate the caller's dict
            order["validate"] = "true"
            print(f"[SANDBOX] Order submitted with validate=true (no fill): "
                  f"{order.get('pair')} {order.get('type')} {order.get('volume')}")

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

        Paper mode: logs the order and returns a mock success (no HTTP call).
        Live mode: calls _kraken_futures_private → /sendorder on futures.kraken.com.
        Returns the result dict on success, or None on failure.
        """
        if not order:
            return None

        if self.futures_paper_mode:
            paper_id = f"PAPER-{int(time.time() * 1000) % 10_000_000}"
            print(f"[PAPER FUTURES FILLED]  {order['symbol']}"
                  f"  side={order['side']}  size={order['size']}"
                  f"  type={order['orderType']}  id={paper_id}")
            return {"status": "paperFilled", "order_id": paper_id}

        resp = self._kraken_futures_private("/sendorder", order)
        if not resp:
            # Soft error — log and skip rather than kill switch
            print("[LIVE FUTURES] No response from /sendorder — skipping this order")
            return None

        result_status = resp.get("result", "")
        error_msg     = resp.get("error", "")
        if result_status != "success" or error_msg:
            print(f"[LIVE FUTURES REJECTED] status={result_status!r}  error={error_msg!r}"
                  f"  order={order}")
            # Auth failure → futures credentials are wrong / not enabled.
            # Disable the overlay for the rest of the session so the spot bot
            # can continue running uninterrupted.  No kill switch.
            if "authenticationError" in str(error_msg) or "invalidCredentials" in str(error_msg):
                print("[LIVE FUTURES] Auth failed — futures overlay disabled for this session.")
                print("  Kraken Futures requires separate credentials from futures.kraken.com")
                self.futures_available = False
            return None

        send_status = resp.get("sendStatus", {})
        order_id    = send_status.get("order_id") or send_status.get("orderId")
        print(f"[LIVE FUTURES SUBMITTED] order_id={order_id}  status={send_status.get('status')}")
        return send_status

    def _kraken_futures_private(self, path: str, data: dict | None = None,
                                method: str = "POST"):
        """
        Real Kraken Futures private REST request.

        Kraken Futures signing scheme (different from spot):
          postData  = URL-encoded form body  (empty string for GET requests)
          nonce     = millisecond timestamp string
          message   = SHA-256(postData + nonce + endpoint)
          Authent   = Base64( HMAC-SHA512(message, Base64Decode(api_secret)) )

        Use method="GET" for read-only endpoints (/accounts, /openpositions).
        Use method="POST" for order submission (/sendorder).

        Ref: https://docs.kraken.com/api/docs/futures-api/authenication
        """
        import base64
        import urllib.parse

        if not self.futures_api_key or not self.futures_api_secret:
            self.trigger_kill_switch("Missing Kraken API credentials for Futures")
            return None

        if data is None:
            data = {}

        url      = self.futures_base_url + path
        nonce    = str(int(time.time() * 1000))
        postdata = urllib.parse.urlencode(data) if data else ""

        # 1. SHA-256 of (postData + nonce + endpoint)
        sha256_hash = hashlib.sha256(
            (postdata + nonce + path).encode()
        ).digest()

        # 2. HMAC-SHA512 using base64-decoded secret
        try:
            decoded_secret = base64.b64decode(self.futures_api_secret)
        except Exception:
            decoded_secret = self.futures_api_secret.encode()

        signature = base64.b64encode(
            hmac.new(decoded_secret, sha256_hash, hashlib.sha512).digest()
        ).decode()

        headers = {
            "APIKey":       self.futures_api_key,
            "Nonce":        nonce,
            "Authent":      signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=10)
            else:
                resp = requests.post(url, headers=headers, data=postdata, timeout=10)
        except Exception as e:
            print(f"[FUTURES REQUEST ERROR] {e}")
            return None

        if resp.status_code != 200:
            print(f"[FUTURES HTTP ERROR] {resp.status_code}  body={resp.text[:200]}")
            return None

        try:
            return resp.json()
        except Exception:
            print("[FUTURES] JSON decode error")
            return None

    def fetch_futures_wallet(self) -> float:
        """
        Fetch the USD collateral balance from the Kraken Futures account.

        Returns 0.0 immediately when ENABLE_FUTURES is False — no HTTP call
        is made and no futures state is mutated.

        Paper mode: simulates 25 % of current spot equity as collateral.
        Live mode: calls GET /accounts on futures.kraken.com.

        Updates self.futures_wallet_usd and self.last_futures_wallet_ts.
        """
        if not self.ENABLE_FUTURES:
            return 0.0   # hard gate — futures disabled

        if not self.futures_available:
            return self.futures_wallet_usd

        if self.futures_paper_mode:
            spot_equity = self.compute_total_equity()
            balance = round(spot_equity * 0.25, 2)
            self.futures_wallet_usd     = balance
            self.last_futures_wallet_ts = time.time()
            print(f"[PAPER FUTURES WALLET] Simulated collateral: ${balance:.2f}"
                  f"  (25% of spot equity ${spot_equity:.2f})")
            return balance

        resp = self._kraken_futures_private("/accounts", {}, method="GET")
        if not resp:
            print("[FUTURES WALLET] Could not fetch balance — keeping last known:"
                  f" ${self.futures_wallet_usd:.2f}")
            return self.futures_wallet_usd

        # Detect API-level auth errors (returns 200 OK but result="error")
        if resp.get("result") == "error":
            err = resp.get("error", "unknown")
            print(f"[FUTURES WALLET] API error: {err}")
            if "auth" in str(err).lower() or "credential" in str(err).lower():
                print("[FUTURES WALLET] Auth failed — futures overlay disabled.")
                print("  Add KRAKEN_FUTURES_API_KEY + KRAKEN_FUTURES_API_SECRET secrets")
                print("  from: https://futures.kraken.com → Settings → API Keys")
                self.futures_available = False
            return self.futures_wallet_usd

        # Kraken Futures /accounts returns:
        # { "result": "success", "accounts": { "flex": {...}, "cash": {...} } }
        # The flex account is the multi-collateral pool; cash is USD-only.
        accounts = resp.get("accounts", {})
        print(f"[FUTURES WALLET] Account keys: {list(accounts.keys())}")

        # Try flex account first (multi-collateral), then cash account
        flex = accounts.get("flex", {})
        cash = accounts.get("cash", {})

        balance = 0.0
        if flex:
            # portfolioValue is the USD-equivalent total value
            balance = float(flex.get("portfolioValue", 0.0))
        if balance == 0.0 and cash:
            balance = float(cash.get("balance", 0.0))

        # Fallback: sum all account balances if neither key found
        if balance == 0.0 and accounts:
            for acct_name, acct_data in accounts.items():
                if isinstance(acct_data, dict):
                    b = float(acct_data.get("portfolioValue",
                              acct_data.get("balance", 0.0)))
                    balance = max(balance, b)

        self.futures_wallet_usd     = balance
        self.last_futures_wallet_ts = time.time()
        print(f"[FUTURES WALLET] Balance: ${balance:.2f} USD collateral")
        return balance

    # ------------------------------------------------------------------
    # Live order execution (overrides SimulatedBroker private helpers)
    # Signature must match the parent: (asset, price, delta_exposure, microstructure_fn)
    # ------------------------------------------------------------------

    def _execute_spot_trade(self, asset, price, delta_exposure, microstructure_fn=None):
        """
        Live spot execution override.

        delta_exposure here is a FRACTIONAL exposure delta (from execute_portfolio_exposure).
        This path converts the fraction to coin units via _fractional_to_coin_units()
        before building the order payload — fixing the previous ambiguity where
        `volume` received a raw fraction instead of coin units.

        Returns the estimated USD fee so the parent can deduct it from
        agent.balance and agent.realized_pnl (fixes the bug where this always
        returned 0.0 and the agent's internal balance was never reduced by fees).
        """
        equity = self.compute_total_equity() or 1.0
        coin_units   = self._fractional_to_coin_units(delta_exposure, equity, price)
        notional_usd = abs(delta_exposure) * equity
        side         = "buy" if delta_exposure > 0 else "sell"

        if not self._pre_trade_safety(asset, price, notional_usd):
            print(f"[LIVE SPOT BLOCKED SAFETY] {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if coin_units <= 0:
            print(f"[LIVE SPOT NO-OP]     {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        order = self._build_spot_order(asset, price, coin_units, side)
        if order is None:
            print(f"[LIVE SPOT NO-OP]     {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        # Estimated fee — returned so caller can update agent.balance
        fee_usd = notional_usd * self.taker_fee

        if self.dry_run:
            print(f"[LIVE SPOT DRY-RUN]   {asset}  {side}  {coin_units:.6f} coins"
                  f"  notional=${notional_usd:.2f}  fee≈${fee_usd:.4f}"
                  f"  order={order}")
            return fee_usd   # return estimated cost even in dry-run so balance is realistic

        result = self._submit_spot_order(order)
        if result is None:
            print(f"[LIVE SPOT FAILED]    {asset}  order={order}")
            return 0.0

        self.spot_positions[asset] = (
            self.spot_positions.get(asset, 0.0)
            + (coin_units if side == "buy" else -coin_units)
        )
        self.cumulative_fees_usd += fee_usd
        print(f"[LIVE SPOT FILLED] {side.upper()} {coin_units:.6f} {asset}"
              f"  @ ${price:.4f}  notional=${notional_usd:.2f}  fee≈${fee_usd:.4f}")
        return fee_usd

    def _execute_futures_trade(self, asset, price, delta_exposure, microstructure_fn=None):
        """
        Live futures execution override.

        Fully suppressed when ENABLE_FUTURES is False (US-user safety gate).
        delta_exposure is a FRACTIONAL delta (fraction of equity).
        Safety checks use equity-based USD notional (not coin × price).
        Adds the missing daily-loss check that the previous version omitted.
        """
        if not self.ENABLE_FUTURES:
            return 0.0   # silently suppressed — futures are disabled

        if not self.check_health():
            return 0.0

        if not self._check_trade_rate():
            return 0.0

        if not self._check_daily_loss():
            return 0.0

        equity       = self.compute_total_equity() or 1.0
        usd_notional = abs(delta_exposure) * equity

        if usd_notional > self.max_notional_per_asset:
            print(f"[SAFETY] Futures notional cap: {asset}"
                  f"  frac={delta_exposure:+.4f} × equity={equity:.2f}"
                  f" = ${usd_notional:.2f} > ${self.max_notional_per_asset}")
            return 0.0

        order = self._build_futures_order(asset, price, delta_exposure)
        if order is None:
            label = "PAPER FUTURES NO-OP " if self.futures_paper_mode else "LIVE FUTURES NO-OP "
            print(f"[{label}]  {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if self.dry_run:
            label = "PAPER FUTURES DRY-RUN" if self.futures_paper_mode else "LIVE FUTURES DRY-RUN"
            print(f"[{label}] {asset}  order={order}")
            return 0.0

        result = self._submit_futures_order(order)
        if result is None and not self.futures_paper_mode:
            print(f"[LIVE FUTURES FAILED]  {asset}  order={order}")
        return 0.0

    def run_futures_overlay(self, agent, prices, regime=None):
        """
        Futures-only pass — mirrors the futures section of
        execute_portfolio_exposure() but never touches spot positions.

        Called from the live RL agent after spot orders have been placed,
        so the spot core stays exactly as filled and the futures layer adds
        a tactical / hedging overlay on top.

        Parameters
        ----------
        agent   : RLAgent — must have .assets, .target_exposures,
                  .max_long, .max_short
        prices  : dict {asset: float}  — live prices
        regime  : dict with optional keys:
                    cycle_phase (int, 0-3),  vol_scaler (float),
                    panic_risk  (int, 0-2),  top_risk   (int, 0-2)
                  Omitted keys default to safe neutral values.
        """
        if not self.ENABLE_FUTURES:
            return   # hard gate — futures fully disabled for US users

        if not self.check_health():
            return

        if not self.futures_available:
            return

        # Refresh futures wallet balance (throttled: at most once per 5 min)
        if time.time() - self.last_futures_wallet_ts > 300:
            self.fetch_futures_wallet()

        # If no collateral is known yet, try once and bail if still zero
        if self.futures_wallet_usd <= 0:
            self.fetch_futures_wallet()
            if self.futures_wallet_usd <= 0:
                print("[FUT OVERLAY] No collateral in futures wallet — skipping overlay")
                return

        regime      = regime or {}
        cycle_phase = int(regime.get("cycle_phase", 1))   # 1 = late-accum / early-bull
        vol_scaler  = float(regime.get("vol_scaler",  1.0))
        panic_risk  = int(regime.get("panic_risk",    0))
        top_risk    = int(regime.get("top_risk",      0))

        spot_w, fut_w = _get_spot_futures_weights(cycle_phase)

        # Total USD notional the futures wallet can support (collateral × leverage).
        # We keep per-asset allocation modest to stay within margin requirements.
        futures_budget_usd  = self.futures_wallet_usd * self.futures_max_leverage
        n_tradeable         = len(self.kraken_futures_pairs)   # assets with liquid contracts
        per_asset_budget    = futures_budget_usd / max(n_tradeable, 1)

        mode_tag = "PAPER FUT OVERLAY" if self.futures_paper_mode else "FUT OVERLAY"
        print(f"[{mode_tag}] wallet=${self.futures_wallet_usd:.2f}"
              f"  budget=${futures_budget_usd:.2f}"
              f"  per_asset=${per_asset_budget:.2f}"
              f"  cycle={cycle_phase}  vol_scaler={vol_scaler:.2f}"
              f"  panic={panic_risk}  top={top_risk}")

        for a in agent.assets:
            # Re-check after each asset in case auth failure disabled futures mid-loop
            if not self.futures_available:
                break

            # Skip assets that have no Kraken Futures perpetual contract
            if a not in self.kraken_futures_pairs:
                continue

            price = prices.get(a, 0.0)
            if price <= 0:
                continue

            # Lazy-init per-asset state
            if a not in self.futures_positions:
                self.futures_positions[a] = 0.0
                self.funding_rates[a]     = 0.0
            if a not in self.spot_positions:
                self.spot_positions[a] = 0.0

            cap_long  = agent.max_long.get(a,  1.0)
            cap_short = agent.max_short.get(a, -1.0)

            raw_target = max(cap_short,
                             min(cap_long, agent.target_exposures.get(a, 0.0)))

            # Per-asset beta tilt on top of cycle-phase weights
            _spot_w, _fut_w = _adjust_weights_for_asset(a, spot_w, fut_w)
            fut_target = raw_target * _fut_w

            # Global vol scaling
            fut_target *= vol_scaler

            # Per-asset volatility targeting (EMA of |return|)
            # In live mode we don't have per-step returns, so we feed 0.0;
            # existing EMA history (built from previous cycles) still applies.
            self.update_asset_volatility(a, 0.0)
            _a_vol     = self.asset_vol_ema.get(a, 0.0)
            _vol_scale = max(0.5, min(1.5, 1.0 / (1.0 + 5.0 * _a_vol)))
            fut_target *= _vol_scale

            if _a_vol > 0.05:
                print(f"  [FUT OVERLAY VolTarget] {a}: vol={_a_vol:.4f}"
                      f"  scale={_vol_scale:.3f}")

            # Panic de-risking
            if panic_risk == 1:
                fut_target *= 0.5
            elif panic_risk == 2:
                fut_target *= 0.1

            # Blow-off top trim
            if top_risk == 1:
                fut_target *= 0.7
            elif top_risk == 2:
                fut_target *= 0.3

            # Funding-aware adjustment
            _raw_funding = self.funding_rates.get(a, 0.0)
            self.update_funding_rate(a, _raw_funding)
            funding     = self.funding_ema.get(a, 0.0)
            funding_adj = max(0.5, min(1.5, 1.0 - 0.25 * funding))
            fut_target *= funding_adj

            if abs(funding) > 0.05:
                print(f"  [FUT OVERLAY Funding] {a}: ema={funding:.4f}"
                      f"  adj={funding_adj:.3f}")

            # Clamp within per-asset caps
            fut_target = max(cap_short, min(cap_long, fut_target))

            # Futures-only hedging on severe panic / blow-off top
            if self.enable_futures_hedging and (panic_risk == 2 or top_risk == 2):
                net_long = self.spot_positions.get(a, 0.0) + max(fut_target, 0.0)
                if net_long > 0:
                    fut_target = -min(self.max_hedge_ratio * net_long, net_long)
                    print(f"  [FUT OVERLAY Hedge] {a}: net_long={net_long:.3f}"
                          f"  hedge={fut_target:.3f}")

            current_fut = self.futures_positions[a]
            delta_fut   = fut_target - current_fut

            # 1. Per-asset futures cap (40 % of equity)
            fut_target = max(-self.max_asset_futures,
                             min(self.max_asset_futures, fut_target))
            delta_fut  = fut_target - current_fut

            # 2. Total futures notional cap
            _ts, _tf, _te = self._compute_portfolio_exposure()
            if _tf + abs(delta_fut) > self.max_futures_notional:
                _allowed  = max(0.0, self.max_futures_notional - _tf)
                delta_fut = (min(delta_fut, _allowed) if delta_fut > 0
                             else max(delta_fut, -_allowed))
                fut_target = current_fut + delta_fut

            # 3. Total portfolio leverage cap (spot + futures)
            if _te + abs(delta_fut) > self.max_portfolio_leverage:
                _allowed  = max(0.0, self.max_portfolio_leverage - _te)
                delta_fut = (min(delta_fut, _allowed) if delta_fut > 0
                             else max(delta_fut, -_allowed))

            asset_tag = "PAPER FUT" if self.futures_paper_mode else "FUT OVERLAY"
            if abs(delta_fut) > 0.01:
                delta_fut_slipped = self._apply_slippage(delta_fut,
                                                         self.futures_slippage_bps)
                print(f"  [{asset_tag}] {a}: delta={delta_fut:+.4f}"
                      f"  slipped={delta_fut_slipped:+.4f}  price={price}"
                      f"  pos={current_fut:+.4f} → {current_fut + delta_fut_slipped:+.4f}")
                self._execute_futures_trade(a, price, delta_fut_slipped)
                self.futures_positions[a] = current_fut + delta_fut_slipped
            else:
                print(f"  [{asset_tag}] {a}: delta={delta_fut:+.5f}  (below 1% — skip)")

    # ------------------------------------------------------------------
    # ETF hedging overlay
    # ------------------------------------------------------------------

    def run_etf_overlay(self, agent, prices, regime=None):
        """
        ETF hedging pass — executes ETHD / SETH spot orders to create a
        leveraged long or short Ethereum hedge without using futures.

        This method replaces the disabled futures overlay for US users.
        It is called from the RL agent after all spot crypto orders have
        been placed, so the crypto core is never disturbed.

        Behaviour
        ---------
        • ETF orders are gated by MarketHours:
            – Regular hours (09:30-16:00 ET): market orders
            – Pre/after-market: limit orders
            – Closed (overnight/weekends): no orders
        • Combined |ETHD| + |SETH| allocation ≤ 30 % of total equity.
        • ETF positions are stored in self.etf_positions (separate from
          self.spot_positions so the 24/7 crypto layer is unaffected).

        Parameters
        ----------
        agent  : RLAgent — provides .assets, .target_exposures,
                 .max_long, .max_short (used for regime inference)
        prices : dict {asset: float} — live prices for all tracked assets
        regime : optional dict with keys:
                   cycle_phase (int 0-3), vol_scaler (float),
                   panic_risk  (int 0-2), top_risk   (int 0-2)
                 If None, a neutral regime is assumed.
        """
        if not self.check_health():
            return

        if not self._check_daily_loss():
            return

        if not self._market_hours.etf_trading_allowed():
            print(f"[ETF OVERLAY] {self._market_hours.status_line()} — skipping")
            return

        regime = regime or {}
        equity = self.compute_total_equity()
        if equity <= 0:
            return

        # Build ETF price dict — fetch from live_prices or prices argument
        etf_prices = {}
        for a in ETF_ASSETS:
            p = self.live_prices.get(a) or prices.get(a, 0.0)
            if p > 0:
                etf_prices[a] = p

        if not etf_prices:
            print("[ETF OVERLAY] No ETF prices available — skipping")
            return

        # Check 30 % cap before computing new orders
        if self.etf_hedger.cap_breached(equity, self.etf_positions, etf_prices):
            print(
                f"[ETF OVERLAY] 30% cap already breached "
                f"(frac={self.etf_hedger.etf_portfolio_fraction(equity, self.etf_positions, etf_prices):.3f}) "
                "— skipping"
            )
            return

        orders = self.etf_hedger.compute_orders(
            regime        = regime,
            equity        = equity,
            etf_prices    = etf_prices,
            etf_positions = self.etf_positions,
        )

        for order in orders:
            asset   = order["asset"]
            side    = order["side"]
            units   = order["units"]
            notional = order["notional"]

            if not self._pre_trade_safety(asset, etf_prices.get(asset, 0.0), notional):
                continue

            payload = self._build_spot_order(
                asset      = asset,
                price      = etf_prices.get(asset, 0.0),
                coin_units = units,
                side       = side,
            )
            if payload is None:
                continue

            if self.dry_run:
                print(f"  [ETF DRY-RUN] {side.upper()} {units:.6f} {asset}"
                      f"  notional=${notional:.2f}  payload={payload}")
            else:
                self._submit_spot_order(payload)

            # Update internal ETF position tracking
            if side == "buy":
                self.etf_positions[asset] = self.etf_positions.get(asset, 0.0) + units
            else:
                current_held = self.etf_positions.get(asset, 0.0)
                if units > current_held:
                    print(
                        f"  [ETF WARNING] Sell {units:.6f} {asset} exceeds"
                        f" tracked holdings {current_held:.6f} — capping to held qty"
                    )
                    units = current_held
                self.etf_positions[asset] = current_held - units

            print(
                f"  [ETF OVERLAY] {side.upper()} {units:.6f} {asset}"
                f"  notional=${notional:.2f}"
                f"  pos_after={self.etf_positions[asset]:.6f}"
            )



# ---------------------------------------------------------------------------
# PaperBroker — full synthetic execution layer for Krakbot sandbox mode
# ---------------------------------------------------------------------------

class PaperBroker(LiveBroker):
    """
    Fully simulated execution layer for Krakbot sandbox mode.

    PaperBroker mirrors the LiveBroker interface exactly so the agent, strategy,
    and main loop can call it without any code changes.  Execution is simulated
    locally via a synthetic fill model:

        fill_price = mid ×  (1 + slippage)   for buys
        fill_price = mid ×  (1 − slippage)   for sells

    Fees use the same Kraken taker schedule as LiveBroker (0.40% retail tier).

    Real Kraken HTTP calls are still made for:
        · Price data     — /0/public/Ticker (batched, 1 call per refresh)
        · No order calls — all submission paths are short-circuited

    Every synthetic fill is:
        · Appended to self.paper_trade_history (in-memory list)
        · Written to project/logs/paper_trades.csv (structured log)
        · Reflected immediately in paper_cash, paper_positions, and
          paper_realized_pnl so the agent receives realistic feedback

    Configuration
    -------------
    USE_PAPER_BROKER   env var — toggle in Replit Secrets (default "true")
    PAPER_SLIPPAGE     env var — override default 0.05% (e.g. "0.001" = 0.1%)
    """

    DEFAULT_SLIPPAGE: float = 0.0005   # 0.05% — realistic top-of-book crypto

    # Log path — relative to this file's directory
    LOG_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "logs", "paper_trades.csv"
    )

    def __init__(self, initial_cash: float, slippage: float | None = None,
                 trade_archive=None, **kwargs):
        """
        Parameters
        ----------
        initial_cash  : Starting USD balance for the paper account.
        slippage      : Fractional slippage per fill.  Defaults to
                        PAPER_SLIPPAGE env var → DEFAULT_SLIPPAGE (0.05%).
        trade_archive : Optional TradeArchive instance for persistent SQLite
                        logging.  When provided, every fill is recorded via
                        archive.record_trade().
        **kwargs      : Forwarded to LiveBroker.__init__.
        """
        # LiveBroker must be in dry_run=False so its monitoring infrastructure
        # (kill switch, health checks, price fetching) is fully armed.
        # PaperBroker short-circuits order submission before any HTTP write
        # reaches Kraken — so dry_run=False here is safe.
        kwargs.setdefault("dry_run", False)
        super().__init__(**kwargs)

        if slippage is None:
            slippage = float(os.getenv("PAPER_SLIPPAGE", str(self.DEFAULT_SLIPPAGE)))
        self.paper_slippage = slippage

        # --- Paper account state ------------------------------------------
        self._initial_cash         = float(initial_cash)  # preserved for EOD report
        self.paper_cash            = float(initial_cash)
        self.paper_positions: dict = {}    # asset → coin quantity (float)
        self.paper_cost_basis: dict = {}   # asset → weighted avg cost (USD/coin)
        self.paper_realized_pnl    = 0.0
        self.paper_cumulative_fees = 0.0
        self.paper_trade_history: list = []

        # --- SQLite trade archive (optional) ------------------------------
        self._trade_archive = trade_archive  # TradeArchive | None
        # Active strategy name — updated by set_strategy_name() so the
        # archive can attribute each fill to the agent that generated it.
        self._current_strategy: str = ""

        # --- CSV log setup ------------------------------------------------
        import csv as _csv
        log_path = os.path.abspath(self.LOG_PATH)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._csv_file   = open(log_path, "w", newline="", buffering=1)
        self._csv_writer = _csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "asset", "side", "size_coins", "fill_price",
            "notional_usd", "fee_usd", "realized_pnl_usd", "position_after_trade",
        ])
        self._csv_file.flush()

        # --- Scale safety caps to paper account size ----------------------
        # LiveBroker's first-night harness defaults ($50/trade, $200 total)
        # are sized for small real-money accounts, not a paper simulation.
        # Scale to realistic multiples of the paper starting cash so that
        # BullBearRotationalTrader can take normal-sized positions (e.g.
        # 10-20% of $10,000 = $1,000-$2,000 per order).
        self.max_notional_per_asset = initial_cash * 0.35   # 35% max per single order
        self.max_total_notional     = initial_cash * 2.0    # 2× equity max total exposure

        print(
            f"[PaperBroker] INITIALIZED\n"
            f"  starting_cash          = ${initial_cash:.2f}\n"
            f"  slippage               = {self.paper_slippage:.4%}\n"
            f"  max_notional_per_asset = ${self.max_notional_per_asset:.2f}\n"
            f"  max_total_notional     = ${self.max_total_notional:.2f}\n"
            f"  trade_log              = {log_path}\n"
            f"  archive                = {'enabled (' + self._trade_archive.db_path + ')' if self._trade_archive else 'disabled'}\n"
            f"  All fills are synthetic — no real orders sent to Kraken."
        )

    # ------------------------------------------------------------------
    # Equity and positions — paper-authoritative (ignores Kraken balances)
    # ------------------------------------------------------------------

    def set_strategy_name(self, name: str) -> None:
        """
        Set the active strategy name for archive attribution.

        Call this before (or immediately after) each group of fills so the
        archive can tag trades with the agent/strategy that generated them.
        """
        self._current_strategy = name

    def compute_total_equity(self) -> float:
        """
        Paper equity = paper_cash + all paper positions at current live prices.
        This is the authoritative equity source for all safety checks,
        daily-loss limits, and agent balance feedback.
        """
        equity = self.paper_cash
        for asset, qty in self.paper_positions.items():
            price = self.live_prices.get(asset, 0.0)
            equity += qty * price
        return equity

    def get_unrealized_pnl(self) -> float:
        """
        Mark-to-market unrealized PnL across all open positions.
        = sum over assets of (current_price − avg_cost_basis) × qty
        """
        total = 0.0
        for asset, qty in self.paper_positions.items():
            price = self.live_prices.get(asset, 0.0)
            cost  = self.paper_cost_basis.get(asset, 0.0)
            total += (price - cost) * qty
        return total

    def get_position_summary(self) -> dict:
        """
        Per-asset snapshot: qty, avg_cost, current price, mtm value,
        unrealized PnL, and fractional equity exposure.
        Useful for periodic health logs and dashboards.
        """
        equity = self.compute_total_equity()
        summary = {}
        for asset, qty in self.paper_positions.items():
            if qty == 0.0:
                continue
            price = self.live_prices.get(asset, 0.0)
            cost  = self.paper_cost_basis.get(asset, 0.0)
            mtm   = qty * price
            summary[asset] = {
                "qty":      round(qty, 8),
                "avg_cost": round(cost, 6),
                "price":    round(price, 6),
                "mtm_usd":  round(mtm, 4),
                "upnl_usd": round((price - cost) * qty, 4),
                "exposure": round(mtm / equity, 4) if equity > 0 else 0.0,
            }
        return summary

    # ------------------------------------------------------------------
    # Account sync — skip Kraken balance API; fetch prices only
    # ------------------------------------------------------------------

    def sync_live_account_state(self):
        """
        Paper override: fetch live prices from Kraken (needed for PnL and
        fill calculations), but use internal paper state for balances.

        Returns (pseudo_balances, {}) on success, None on price-fetch failure.
        The pseudo_balances dict mirrors the Kraken balance-key format so the
        main loop can log it without modification.
        """
        if not self.check_health():
            print("[PAPER SYNC BLOCKED] Kill-switch active — skipping")
            return None

        print("[PAPER SYNC] Fetching live prices from Kraken...")
        self.fetch_live_prices()

        if not self.live_prices:
            self.trigger_kill_switch(
                "Paper sync failed — could not fetch any live prices from Kraken"
            )
            return None

        # Build pseudo-balance dict (mirrors Kraken key names)
        pseudo = {"ZUSD": str(round(self.paper_cash, 4))}
        for asset, qty in self.paper_positions.items():
            bal_key = self.kraken_balance_keys.get(asset, asset)
            pseudo[bal_key] = str(round(qty, 8))

        equity = self.compute_total_equity()
        upnl   = self.get_unrealized_pnl()
        print(
            f"[PAPER SYNC] equity=${equity:.2f}  cash=${self.paper_cash:.2f}"
            f"  upnl=${upnl:+.2f}  rpnl=${self.paper_realized_pnl:+.2f}"
            f"  fees=${self.paper_cumulative_fees:.4f}"
            f"  prices={len(self.live_prices)} assets"
        )
        return pseudo, {}

    # ------------------------------------------------------------------
    # Core fill engine — synthetic execution with slippage + fees
    # ------------------------------------------------------------------

    def _paper_fill(self, asset: str, side: str, coin_units: float,
                    mid_price: float) -> float:
        """
        Execute one synthetic fill.

        Fill model:
            buy  → fill_price = mid × (1 + slippage)  cash decreases
            sell → fill_price = mid × (1 - slippage)  cash increases

        Fee = notional × taker_fee (same schedule as LiveBroker).
        Realized PnL on sells = (fill_price - avg_cost) × qty_sold - fee.

        Updates:
            paper_cash, paper_positions, paper_cost_basis,
            paper_realized_pnl, paper_cumulative_fees,
            cumulative_fees_usd (shared with LiveBroker)

        Appends one row to paper_trade_history and writes to the CSV log.

        Parameters
        ----------
        asset      : internal name ("SOL", "BTC", …)
        side       : "buy" or "sell"
        coin_units : unsigned quantity in base-asset units (> 0)
        mid_price  : current mid-price in USD

        Returns
        -------
        float — fee_usd charged for this fill
        """
        if coin_units <= 0 or mid_price <= 0:
            return 0.0

        # Apply slippage
        if side == "buy":
            fill_price = mid_price * (1.0 + self.paper_slippage)
        else:
            fill_price = mid_price * (1.0 - self.paper_slippage)

        notional_usd = coin_units * fill_price
        fee_usd      = notional_usd * self.taker_fee

        # Current position state
        cur_qty  = self.paper_positions.get(asset, 0.0)
        cur_cost = self.paper_cost_basis.get(asset, 0.0)
        realized_pnl = 0.0

        if side == "buy":
            new_qty = cur_qty + coin_units
            # Weighted-average cost update
            if new_qty > 0:
                self.paper_cost_basis[asset] = (
                    (cur_qty * cur_cost + coin_units * fill_price) / new_qty
                )
            self.paper_positions[asset] = new_qty
            # Cash decreases by notional + fee
            self.paper_cash -= (notional_usd + fee_usd)

        else:  # sell
            # Only sell what we actually hold (clamp at zero for spot-only)
            sell_qty = min(coin_units, max(cur_qty, 0.0))
            if sell_qty < coin_units:
                print(f"[PAPER FILL] WARNING: tried to sell {coin_units:.6f} {asset}"
                      f" but only hold {cur_qty:.6f} — clamped to {sell_qty:.6f}")
                coin_units = sell_qty
                notional_usd = coin_units * fill_price
                fee_usd      = notional_usd * self.taker_fee

            realized_pnl = (fill_price - cur_cost) * coin_units - fee_usd
            self.paper_realized_pnl    += realized_pnl
            self.paper_positions[asset] = cur_qty - coin_units
            # Cash increases by proceeds minus fee
            self.paper_cash += (fill_price * coin_units) - fee_usd

        # Sync fees to the shared LiveBroker counter
        self.paper_cumulative_fees += fee_usd
        self.cumulative_fees_usd   += fee_usd

        # Keep spot_positions in sync so execute_portfolio_exposure logic works
        self.spot_positions[asset] = self.paper_positions.get(asset, 0.0)

        position_after = self.paper_positions.get(asset, 0.0)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # In-memory record
        record = {
            "timestamp":            ts,
            "asset":                asset,
            "side":                 side,
            "size_coins":           round(coin_units, 8),
            "fill_price":           round(fill_price, 6),
            "notional_usd":         round(notional_usd, 4),
            "fee_usd":              round(fee_usd, 6),
            "realized_pnl_usd":     round(realized_pnl, 6),
            "position_after_trade": round(position_after, 8),
        }
        self.paper_trade_history.append(record)

        # CSV row (line-buffered — buffering=1 means each row flushes immediately)
        self._csv_writer.writerow([
            record["timestamp"],   record["asset"],          record["side"],
            record["size_coins"],  record["fill_price"],     record["notional_usd"],
            record["fee_usd"],     record["realized_pnl_usd"], record["position_after_trade"],
        ])

        # SQLite archive (optional) — non-fatal if unavailable
        if self._trade_archive is not None:
            try:
                self._trade_archive.record_trade(
                    record, strategy_name=self._current_strategy
                )
            except Exception as _exc:  # noqa: BLE001
                print(f"[TradeArchive] WARNING: failed to record trade: {_exc}")

        print(
            f"[PAPER FILL] {side.upper():4s} {coin_units:.6f} {asset}"
            f"  mid=${mid_price:.4f}  fill=${fill_price:.4f}"
            f"  notional=${notional_usd:.2f}  fee=${fee_usd:.4f}"
            f"  rpnl=${realized_pnl:+.4f}  pos={position_after:.6f}"
            f"  cash=${self.paper_cash:.2f}  equity=${self.compute_total_equity():.2f}"
        )
        return fee_usd

    # ------------------------------------------------------------------
    # Execution overrides — route to _paper_fill instead of Kraken HTTP
    # ------------------------------------------------------------------

    def _execute_spot_trade(self, asset, price, delta_exposure,
                            microstructure_fn=None) -> float:
        """
        Paper execution for the execute_portfolio_exposure() path.

        delta_exposure is a signed fractional exposure delta (from the RL agent).
        Converts to coin units via _fractional_to_coin_units() and routes to
        _paper_fill() — matching the convention LiveBroker uses.

        Returns fee_usd so the parent can update agent.balance and realized_pnl.
        """
        if not self.check_health():
            return 0.0

        equity     = self.compute_total_equity() or 1.0
        coin_units = self._fractional_to_coin_units(delta_exposure, equity, price)
        notional   = abs(delta_exposure) * equity
        side       = "buy" if delta_exposure > 0 else "sell"

        if not self._pre_trade_safety(asset, price, notional):
            print(f"[PAPER SPOT BLOCKED] {asset}  delta={delta_exposure:+.4f}")
            return 0.0

        if coin_units <= 0:
            return 0.0

        return self._paper_fill(asset, side, coin_units, price)

    def execute_trade(self, symbol: str, side: str, size: float) -> float:
        """
        Paper execution for direct execute_trade() calls (agent.place_order path).

        size is in coin units (positive, unsigned) — same convention as
        LiveBroker.execute_trade.  Returns fee_usd so callers can update
        their internal balance (consistent with LiveBroker's return value).
        """
        price = self.live_prices.get(symbol)
        if not price or price <= 0:
            print(f"[PAPER TRADE SKIPPED] No live price for {symbol}")
            return 0.0

        notional = size * price
        if not self._pre_trade_safety(symbol, price, notional):
            return 0.0

        return self._paper_fill(symbol, side, size, price)

    def _submit_spot_order(self, order: dict):
        """
        Safety no-op — this method should never be reached in paper mode
        because _execute_spot_trade() and execute_trade() both return before
        reaching the submission path.  Logged as a warning if called.
        """
        print(f"[PAPER BROKER] WARNING: _submit_spot_order reached unexpectedly "
              f"(bug — should not happen).  Order: {order}")
        return None

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return a performance snapshot as a dict.

        Keys
        ----
        total_trades, total_buys, total_sells,
        realized_pnl_usd, unrealized_pnl_usd, total_fees_usd,
        win_rate, average_win, average_loss, largest_win, largest_loss,
        current_positions, equity
        """
        trades = self.paper_trade_history

        buys  = [t for t in trades if t["side"] == "buy"]
        sells = [t for t in trades if t["side"] == "sell"]

        # Realised wins and losses come from the realized_pnl_usd on sell fills.
        # Buy records always have realized_pnl_usd == 0 so we can sum over all.
        pnls = [t["realized_pnl_usd"] for t in sells if t["realized_pnl_usd"] != 0.0]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate    = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
        avg_win     = (sum(wins)   / len(wins))   if wins   else 0.0
        avg_loss    = (sum(losses) / len(losses)) if losses else 0.0
        largest_win  = max(wins,   default=0.0)
        largest_loss = min(losses, default=0.0)

        return {
            "total_trades":      len(trades),
            "total_buys":        len(buys),
            "total_sells":       len(sells),
            "realized_pnl_usd":  round(self.paper_realized_pnl, 4),
            "unrealized_pnl_usd": round(self.get_unrealized_pnl(), 4),
            "total_fees_usd":    round(self.paper_cumulative_fees, 4),
            "win_rate":          round(win_rate, 1),
            "average_win":       round(avg_win,  4),
            "average_loss":      round(avg_loss, 4),
            "largest_win":       round(largest_win,  4),
            "largest_loss":      round(largest_loss, 4),
            "current_positions": dict(self.paper_positions),
            "equity":            round(self.compute_total_equity(), 4),
        }

    def print_summary(self) -> None:
        """Print the performance summary in a readable format."""
        s = self.summary()
        positions = s["current_positions"]

        lines = [
            "",
            "=== PAPER TRADING SUMMARY ===",
            f"Total trades: {s['total_trades']} ({s['total_buys']} buys, {s['total_sells']} sells)",
            f"Realized PnL:    {s['realized_pnl_usd']:>+10.2f} USD",
            f"Unrealized PnL:  {s['unrealized_pnl_usd']:>+10.2f} USD",
            f"Total fees:      {s['total_fees_usd']:>10.4f} USD",
            f"Win rate:        {s['win_rate']:>9.1f}%",
            f"Average win:     {s['average_win']:>+10.4f} USD",
            f"Average loss:    {s['average_loss']:>+10.4f} USD",
            f"Largest win:     {s['largest_win']:>+10.4f} USD",
            f"Largest loss:    {s['largest_loss']:>+10.4f} USD",
            f"Equity:          {s['equity']:>10.2f} USD",
            "Positions:",
        ]
        if positions:
            for asset, qty in sorted(positions.items()):
                price  = self.live_prices.get(asset, 0.0)
                value  = qty * price
                lines.append(f"  {asset:<6}  {qty:.6f} coins  ≈ ${value:.2f}")
        else:
            lines.append("  (none)")
        lines.append("=" * 30)
        lines.append("")
        print("\n".join(lines))

    def print_session_summary(self):
        """Print a human-readable PnL and position summary to the console."""
        equity = self.compute_total_equity()
        upnl   = self.get_unrealized_pnl()
        print(
            f"\n{'='*60}\n"
            f"PAPER SESSION SUMMARY\n"
            f"{'='*60}\n"
            f"  Cash (USD):        ${self.paper_cash:>12.4f}\n"
            f"  Unrealized PnL:    ${upnl:>+12.4f}\n"
            f"  Realized PnL:      ${self.paper_realized_pnl:>+12.4f}\n"
            f"  Fees paid:         ${self.paper_cumulative_fees:>12.4f}\n"
            f"  Total equity:      ${equity:>12.4f}\n"
            f"  Trades filled:     {len(self.paper_trade_history)}\n"
            f"{'='*60}"
        )
        summary = self.get_position_summary()
        if summary:
            print("  Open positions:")
            for asset, d in summary.items():
                print(f"    {asset:<6}  qty={d['qty']:.6f}  "
                      f"avg_cost=${d['avg_cost']:.4f}  "
                      f"price=${d['price']:.4f}  "
                      f"mtm=${d['mtm_usd']:.2f}  "
                      f"upnl=${d['upnl_usd']:+.2f}  "
                      f"exposure={d['exposure']:.2%}")
        print()

    def save_eod_report(self) -> str:
        """Write a dated end-of-day analysis report to project/logs/eod_YYYYMMDD.txt.

        The file contains:
          • Session summary (starting capital, final equity, PnL, fees, win rate)
          • Open positions at close
          • Full trade-by-trade log (timestamp, asset, side, size, price, notional, fee, rPnL)
          • Paths to the CSV and SQLite logs for deeper analysis

        Returns the absolute path of the file written.
        """
        import datetime as _dt

        now       = _dt.datetime.now()
        date_str  = now.strftime("%Y%m%d")
        ts_str    = now.strftime("%Y-%m-%d %H:%M:%S")
        log_dir   = os.path.dirname(os.path.abspath(self.LOG_PATH))
        report_path = os.path.join(log_dir, f"eod_{date_str}.txt")

        s = self.summary()

        lines = [
            f"KrakBot — End-of-Day Sandbox Report  ({ts_str})",
            "=" * 60,
            f"  Starting capital:  ${self._initial_cash:>10,.2f}",
            f"  Final equity:      ${s['equity']:>10.2f}",
            f"  Cash remaining:    ${self.paper_cash:>10.2f}",
            f"  Realized PnL:      ${s['realized_pnl_usd']:>+10.2f}",
            f"  Unrealized PnL:    ${s['unrealized_pnl_usd']:>+10.2f}",
            f"  Total fees:        ${s['total_fees_usd']:>10.4f}",
            f"  Net return:        ${(s['equity'] - self._initial_cash):>+10.2f}"
            f"  ({(s['equity'] / self._initial_cash - 1) * 100:>+.2f}%)",
            f"  Total trades:      {s['total_trades']}"
            f"  ({s['total_buys']} buys, {s['total_sells']} sells)",
            f"  Win rate:          {s['win_rate']:.1f}%",
            f"  Average win:       ${s['average_win']:>+10.4f}",
            f"  Average loss:      ${s['average_loss']:>+10.4f}",
            f"  Largest win:       ${s['largest_win']:>+10.4f}",
            f"  Largest loss:      ${s['largest_loss']:>+10.4f}",
            "",
            "  Open Positions at Close:",
        ]

        if s["current_positions"]:
            for asset, qty in sorted(s["current_positions"].items()):
                price = self.live_prices.get(asset, 0.0)
                value = qty * price
                lines.append(
                    f"    {asset:<6}  {qty:.6f} coins"
                    f"  @ ${price:>10.2f}  ≈ ${value:>8.2f}"
                )
        else:
            lines.append("    (none)")

        # Trade-by-trade log
        lines += [
            "",
            "  Trade Log:",
            (
                f"  {'#':>4}  {'Timestamp':19}  {'Asset':6}  {'Side':4}"
                f"  {'Size (coins)':>12}  {'Fill $':>10}  {'Notional':>10}"
                f"  {'Fee':>8}  {'rPnL':>10}"
            ),
            "  " + "-" * 90,
        ]
        for i, t in enumerate(self.paper_trade_history, 1):
            lines.append(
                f"  {i:>4}  {t['timestamp']:19}  {t['asset']:<6}  {t['side']:<4}"
                f"  {t['size_coins']:>12.6f}  {t['fill_price']:>10.4f}"
                f"  {t['notional_usd']:>10.2f}  {t['fee_usd']:>8.4f}"
                f"  {t['realized_pnl_usd']:>+10.4f}"
            )
        if not self.paper_trade_history:
            lines.append("    (no trades executed this session)")

        lines += [
            "",
            "  Log files:",
            f"    CSV trades : {os.path.abspath(self.LOG_PATH)}",
            f"    SQLite DB  : {self._trade_archive.db_path if self._trade_archive else 'N/A'}",
            f"    EOD report : {report_path}",
            "=" * 60,
        ]

        with open(report_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"[PaperBroker] EOD report saved: {report_path}")
        return report_path

    def close(self):
        """Flush and close the CSV trade log file and SQLite archive.  Call at session end."""
        self.print_session_summary()
        self.save_eod_report()
        try:
            self._csv_file.flush()
            self._csv_file.close()
            print(f"[PaperBroker] Trade log closed: {self.LOG_PATH}")
        except Exception:
            pass
        if self._trade_archive is not None:
            try:
                self._trade_archive.close()
                print(f"[PaperBroker] Trade archive closed: {self._trade_archive.db_path}")
            except Exception:
                pass


def run_live_trading_loop(broker, agent, loop_sleep: float = 1.0):
    """
    Main live trading loop.

    Responsibilities per iteration:
    - Respect kill switch (hard stop)
    - Let the agent run one decision step
    - Emit heartbeat + health metrics
    - Run daily rollover
    - Evaluate alert conditions
    - Check for keyboard commands from stdin (non-blocking)
    - Print a performance summary every 15 minutes automatically

    Keyboard commands (type in the workflow console + Enter):
        S  — print PaperBroker performance summary immediately
    """
    import queue
    import threading

    # ---------------------------------------------------------------
    # Non-blocking stdin reader (daemon thread so it dies with main)
    # ---------------------------------------------------------------
    _cmd_queue: queue.Queue = queue.Queue()

    def _stdin_reader():
        try:
            for raw_line in iter(sys.stdin.readline, ""):
                cmd = raw_line.strip().upper()
                if cmd:
                    _cmd_queue.put(cmd)
        except Exception:
            pass   # stdin closed or not a TTY — silently stop

    _stdin_thread = threading.Thread(target=_stdin_reader, daemon=True, name="stdin-reader")
    _stdin_thread.start()

    SUMMARY_INTERVAL_SEC = 900   # 15 minutes
    _last_summary_ts     = time.time()

    def _print_summary_if_paper():
        if isinstance(broker, PaperBroker):
            broker.print_summary()
        else:
            print("[MAIN LOOP] Summary only available in PAPER mode.")

    # Initial sync before trading
    state = broker.sync_live_account_state()
    if state is None:
        print("[MAIN LOOP] Failed initial account sync — aborting")
        return

    balances, positions = state
    print(f"[MAIN LOOP] Initial balances: {balances}")
    print(f"[MAIN LOOP] Initial positions: {positions}")
    print("[MAIN LOOP] Type 'S' + Enter in the console to print a performance summary.")

    try:
        while True:
            # Hard stop if kill switch is active
            if broker.kill_switch:
                print("[MAIN LOOP] Kill switch active — exiting loop")
                break

            # -------------------------------------------------------
            # Keyboard command handler (non-blocking)
            # -------------------------------------------------------
            try:
                cmd = _cmd_queue.get_nowait()
                if cmd == "S":
                    _print_summary_if_paper()
                else:
                    print(f"[MAIN LOOP] Unknown command: {cmd!r}  (known: S)")
            except queue.Empty:
                pass

            # -------------------------------------------------------
            # Automatic 15-minute summary
            # -------------------------------------------------------
            _now = time.time()
            if _now - _last_summary_ts >= SUMMARY_INTERVAL_SEC:
                print("[MAIN LOOP] 15-minute auto-summary:")
                _print_summary_if_paper()
                _last_summary_ts = _now

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
