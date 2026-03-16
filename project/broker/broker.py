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
        # Kraken spot taker fee for standard volume tier (< $50k/month).
        # Maker = 0.16 %, taker = 0.26 %.  Limit orders earn maker rebate when
        # they rest on the book; we default to limit orders now.
        # Round-trip taker cost = 2 × 0.26 % = 0.52 % per trade.
        self.taker_fee           = 0.0026   # 0.26 % taker
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
        # (Kraken prefixes some with X; LINK/HBAR/AVAX/SOL use bare names)
        self.kraken_balance_keys = {
            "BTC":  "XXBT",
            "ETH":  "XETH",
            "SOL":  "SOL",
            "AVAX": "AVAX",
            "LINK": "LINK",
            "HBAR": "HBAR",
            "XRP":  "XXRP",
            "XLM":  "XXLM",
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

        Previous implementation made 8 separate calls (one per asset).  Kraken
        accepts a comma-separated pair list, returning all tickers in one
        response — 8× fewer API calls, much lower rate-limit risk.

        'c' field = last trade closed: [price, lot_volume].
        Returns the result dict and updates self.live_prices on success.
        """
        # Build comma-separated pair string once
        pairs_str = ",".join(self.kraken_pairs.values())
        data = self._kraken_public("/0/public/Ticker", {"pair": pairs_str})

        if not data or "result" not in data:
            print("[PRICE FEED ERROR] Batch ticker request failed or returned no result")
            return {}

        ticker_data = data["result"]
        result      = {}

        # Build a reverse map: Kraken-normalised-pair → internal asset name.
        # Kraken may change the pair key (e.g. XBTUSD → XXBTZUSD) so we match
        # on the pair value we sent, case-insensitively as a fallback.
        pair_to_asset = {v.upper(): k for k, v in self.kraken_pairs.items()}

        for kraken_key, ticker in ticker_data.items():
            asset = pair_to_asset.get(kraken_key.upper())
            if asset is None:
                # Try partial match (e.g. "XXBTZUSD" contains "XBTUSD")
                for sent_pair, a in self.kraken_pairs.items():
                    if sent_pair.upper() in kraken_key.upper() or kraken_key.upper() in sent_pair.upper():
                        asset = sent_pair
                        break
            if asset is None:
                print(f"[PRICE FEED] Unknown pair key from Kraken: {kraken_key!r}")
                continue
            try:
                result[asset] = float(ticker["c"][0])
            except Exception:
                print(f"[PRICE PARSE ERROR] Could not parse last trade for {kraken_key}")

        if result:
            self.live_prices          = result
            self.last_price_timestamp = time.time()
            missing = [a for a in self.kraken_pairs if a not in result]
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
                print(f"[HEARTBEAT] alive at {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        Format a Kraken spot LIMIT-order payload.

        Parameters
        ----------
        asset      : internal asset name (e.g. "SOL")
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
        Limit orders also qualify for maker fee (0.16%) rather than taker (0.26%)
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

        tol = self.limit_order_tolerance
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
        True portfolio value = ZUSD cash + all crypto holdings at live prices.
        Avoids treating buy orders as losses in the daily-loss check.
        """
        balances = self.live_balances or {}
        equity   = float(balances.get("ZUSD", 0.0))
        for asset, bal_key in self.kraken_balance_keys.items():
            qty   = float(balances.get(bal_key, 0.0))
            price = self.live_prices.get(asset, 0.0)
            equity += qty * price
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
