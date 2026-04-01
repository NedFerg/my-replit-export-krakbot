"""
project/broker/alpaca_broker.py
--------------------------------
Alpaca brokerage integration for the trading bot.

Supports:
  • Crypto spot trading (BTC/USD, ETH/USD, SOL/USD, XRP/USD, LINK/USD,
    AVAX/USD, HBAR/USD, XLM/USD) via Alpaca's unified crypto API.
  • ETF / stock trading via Alpaca's standard equity API, using leveraged
    equity ETFs (QLD, TQQQ, SSO, SQQQ, SH) as directional proxies for the
    crypto-market regime-overlay logic that was previously relying on Kraken
    crypto ETPs (ETHU, SLON, XXRP, ETHD, SETH).
  • Pattern Day Trader (PDT) rule compliance — pre-trade checks that block
    orders when the account would breach the 4-day-trade / 5-business-day
    limit in a margin account with < $25 000 equity.
  • Market-hours-aware order routing — market orders during regular session,
    limit orders during extended hours, no ETF orders when market is closed.
  • Paper / live mode switching via ALPACA_BASE_URL environment variable.

Architecture
------------
AlpacaBroker is a standalone class whose public interface mirrors LiveBroker
(broker.py) so it can be dropped in as a replacement when BROKER=alpaca is
set.  The existing ETFHedger (broker/etf_hedging.py) is reused unchanged —
it is exchange-agnostic and just computes what orders to place; AlpacaBroker
handles *how* those orders are routed to Alpaca.

Kraken-specific code in broker.py is untouched and remains available for
BROKER=kraken (the default) or legacy testing.

Configuration (environment variables)
--------------------------------------
  BROKER                 : Set to "alpaca" to use this broker
  ALPACA_API_KEY         : Alpaca API key
  ALPACA_API_SECRET      : Alpaca API secret
  ALPACA_BASE_URL        : API base URL
                           Paper (default): https://paper-api.alpaca.markets
                           Live:            https://api.alpaca.markets
  MAX_ETF_ALLOCATION     : Combined ETF cap as fraction of equity (default 0.30)
  MAX_DAILY_LOSS_PCT     : Session loss cap fraction (default 0.10)
  LIMIT_ORDER_TOLERANCE  : How far from mid price limit orders are placed (default 0.001)
  ALPACA_TAKER_FEE       : Assumed taker fee for crypto (default 0.0015 = 0.15 %)
                           ETF / stock trading on Alpaca is commission-free.
"""

import csv
import os
import time
import logging
from typing import Optional

from utils.market_hours import MarketHours
from broker.etf_hedging import (
    ETFHedger,
    ETFHedgingLayer,
    ETF_ASSETS,
    ALL_ETFS,
    etf_regime_direction as _etf_regime_direction,
    MIN_ORDER_USD as _ETF_MIN_ORDER_USD,
    ETF_MIN_ALLOCATION_USD as _ETF_MIN_ALLOCATION_USD,
    ETF_ORDER_TIMEOUT_SEC as _ETF_ORDER_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset maps
# ---------------------------------------------------------------------------

# Internal ticker → Alpaca crypto pair symbol (24/7 trading)
ALPACA_CRYPTO_PAIRS: dict[str, str] = {
    "BTC":  "BTC/USD",
    "ETH":  "ETH/USD",
    "SOL":  "SOL/USD",
    "XRP":  "XRP/USD",
    "LINK": "LINK/USD",
    "AVAX": "AVAX/USD",
    "HBAR": "HBAR/USD",
    "XLM":  "XLM/USD",
}

# Internal ETF ticker (used by ETFHedger / regime logic) → Alpaca stock ticker
#
# The Kraken crypto ETPs are not available on Alpaca.  Leveraged equity ETFs
# whose underlying indices (NASDAQ-100, S&P 500) are highly correlated with
# crypto market conditions serve as directional proxies:
#
#   ETHU → QLD   ProShares Ultra QQQ          (2× Long  NASDAQ-100)
#   SLON → TQQQ  ProShares UltraPro QQQ       (3× Long  NASDAQ-100)
#   XXRP → SSO   ProShares Ultra S&P 500      (2× Long  S&P 500)
#   ETHD → SQQQ  ProShares UltraPro Short QQQ (3× Short NASDAQ-100) ← main bear hedge
#   SETH → SH    ProShares Short S&P 500      (1× Short S&P 500)    ← lighter hedge
#
ALPACA_ETF_PAIRS: dict[str, str] = {
    "ETHU": "QLD",     # ETH 2× Long  → 2× Long NASDAQ-100
    "SLON": "TQQQ",    # SOL 2× Long  → 3× Long NASDAQ-100
    "XXRP": "SSO",     # XRP 2× Long  → 2× Long S&P 500
    "ETHD": "SQQQ",    # ETH 2× Short → 3× Short NASDAQ-100 (primary bear hedge)
    "SETH": "SH",      # ETH 1× Short → 1× Short S&P 500   (lighter bear hedge)
}

# Reverse map: Alpaca ticker → internal ticker (used when syncing positions)
_ALPACA_ETF_REVERSE: dict[str, str] = {v: k for k, v in ALPACA_ETF_PAIRS.items()}
_ALPACA_CRYPTO_REVERSE: dict[str, str] = {v: k for k, v in ALPACA_CRYPTO_PAIRS.items()}

# Minimum notional size per order accepted by AlpacaBroker (USD)
_MIN_NOTIONAL_USD = float(os.getenv("ALPACA_MIN_NOTIONAL_USD", "1.0"))

# Gain-multiple (× round-trip fee) required before an ETF position is
# closed/rotated.  Mirrors LiveBroker._ETF_FEE_HURDLE_FACTOR.
_ETF_FEE_HURDLE_FACTOR = float(os.getenv("ETF_FEE_HURDLE_FACTOR", "2.5"))


# ---------------------------------------------------------------------------
# AlpacaBroker
# ---------------------------------------------------------------------------

class AlpacaBroker:
    """
    Alpaca brokerage client.

    Drop-in replacement for LiveBroker when BROKER=alpaca is set.
    Reuses ETFHedger unchanged — only the order routing layer differs.
    """

    # Audit log for Alpaca live trades
    LIVE_LOG_PATH: str = os.path.join(
        os.path.dirname(__file__), "..", "logs", "alpaca_trades.csv"
    )

    def __init__(self, dry_run: bool = True):
        self.dry_run     = dry_run
        self.kill_switch = False
        self.last_api_error: Optional[str] = None

        # --- Credentials --------------------------------------------------
        self.api_key    = os.getenv("ALPACA_API_KEY",    "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")
        self.base_url   = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not self.api_key or not self.api_secret:
            logger.warning(
                "[AlpacaBroker] ALPACA_API_KEY / ALPACA_API_SECRET not set — "
                "forcing dry_run=True."
            )
            self.dry_run = True

        # --- Alpaca REST client (lazy import) -----------------------------
        self._api = None
        self._init_alpaca_client()

        # --- Asset maps ---------------------------------------------------
        self.crypto_pairs: dict[str, str] = dict(ALPACA_CRYPTO_PAIRS)
        self.etf_pairs:    dict[str, str] = dict(ALPACA_ETF_PAIRS)

        # Combined lookup: internal ticker → Alpaca symbol
        self._all_pairs: dict[str, str] = {
            **self.crypto_pairs,
            **self.etf_pairs,
        }

        # --- Price / state cache ------------------------------------------
        self.live_prices:           dict[str, float] = {}
        self.last_price_timestamp:  float            = 0.0
        self.live_balances:         dict             = {}
        self.live_positions:        dict[str, float] = {}

        # ETF positions are tracked separately from crypto spot positions
        # so the 30% allocation cap can be enforced independently.
        self.etf_positions:  dict[str, float] = {etf: 0.0 for etf in ALL_ETFS}
        self.spot_positions: dict[str, float] = {}

        # --- ETF hedging layer -------------------------------------------
        self.etf_hedger    = ETFHedger(
            max_etf_allocation=float(os.getenv("MAX_ETF_ALLOCATION", "0.30")),
        )
        self.etf_layer     = ETFHedgingLayer()  # alias; same object under the hood
        self._market_hours = MarketHours()

        # Tracks the most recent confirmed ETF regime direction
        self._prev_etf_regime_dir = "neutral"

        # ETF priority-order pending state
        self._etf_priority_pending   = False
        self._etf_priority_placed_at = 0.0
        self._etf_priority_asset: Optional[str] = None

        # --- Fee accounting -----------------------------------------------
        # Alpaca charges 0% commission on stock/ETF trades.
        # Crypto trading has a ~0.15% fee (configurable).
        self.taker_fee:           float = float(os.getenv("ALPACA_TAKER_FEE", "0.0015"))
        self.maker_fee:           float = 0.0
        self.cumulative_fees_usd: float = 0.0

        # --- Safety limits ------------------------------------------------
        _env_per_asset = os.getenv("MAX_NOTIONAL_PER_ASSET_USD")
        _env_total     = os.getenv("MAX_TOTAL_NOTIONAL_USD")
        self.max_notional_per_asset  = float(_env_per_asset) if _env_per_asset else 50.0
        self.max_total_notional      = float(_env_total)     if _env_total     else 200.0
        self._max_notional_pct       = float(os.getenv("MAX_NOTIONAL_PCT",       "0.35"))
        self._max_total_notional_pct = float(os.getenv("MAX_TOTAL_NOTIONAL_PCT", "2.0"))
        self._per_asset_cap_fixed    = bool(_env_per_asset)
        self._total_cap_fixed        = bool(_env_total)

        self.max_trades_per_hour      = 50
        self._trade_count_window: list[float] = []

        self.max_daily_loss_pct          = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.10"))
        self._max_daily_loss_usd: Optional[float] = None
        self._starting_equity:    Optional[float] = None

        # Rate-gate intervals
        self._health_metrics_interval = int(os.getenv("HEALTH_METRICS_INTERVAL", "60"))
        self._alerting_interval       = int(os.getenv("ALERTING_INTERVAL",       "60"))
        self._last_health_metrics_ts  = 0.0
        self._last_alerting_ts        = 0.0

        # --- Live trade audit log -----------------------------------------
        live_log_path = os.path.abspath(self.LIVE_LOG_PATH)
        os.makedirs(os.path.dirname(live_log_path), exist_ok=True)
        _needs_header = not os.path.exists(live_log_path)
        self._live_csv_file   = open(live_log_path, "a", newline="", buffering=1)
        self._live_csv_writer = csv.writer(self._live_csv_file)
        if _needs_header:
            self._live_csv_writer.writerow([
                "timestamp", "asset", "side", "size", "fill_price",
                "notional_usd", "fee_usd",
            ])
            self._live_csv_file.flush()

        _mode  = "DRY-RUN" if self.dry_run else "LIVE"
        _paper = "paper" if "paper-api" in self.base_url else "LIVE-MONEY"
        logger.info(f"[AlpacaBroker] Initialized ({_mode} | {_paper} endpoint)")
        if not self.dry_run:
            logger.warning(
                "[AlpacaBroker] dry_run=False — LIVE ORDER SUBMISSION ENABLED. "
                "Ensure ALPACA_BASE_URL is correct (paper vs live)."
            )

    # ------------------------------------------------------------------
    # Alpaca REST client
    # ------------------------------------------------------------------

    def _init_alpaca_client(self):
        """Initialize the alpaca_trade_api REST client."""
        if not self.api_key or not self.api_secret:
            return
        try:
            import alpaca_trade_api as tradeapi  # type: ignore[import]
            self._api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version="v2",
            )
            logger.info(f"[AlpacaBroker] Connected to {self.base_url}")
        except ImportError:
            logger.error(
                "[AlpacaBroker] alpaca-trade-api package not installed. "
                "Run: pip install 'alpaca-trade-api>=3.0.0'"
            )
            self._api = None

    # ------------------------------------------------------------------
    # Kill-switch / health
    # ------------------------------------------------------------------

    def trigger_kill_switch(self, reason: str):
        """Immediately halt all order flow."""
        self.kill_switch    = True
        self.last_api_error = reason
        logger.error(f"[ALPACA KILL SWITCH] {reason}")

    def check_health(self) -> bool:
        """Return False when the kill switch is active."""
        return not self.kill_switch

    # ------------------------------------------------------------------
    # Account equity — required interface
    # ------------------------------------------------------------------

    def get_account_equity(self) -> float:
        """
        Return total portfolio equity from Alpaca.

        Falls back to the cached value in self.live_balances when the API
        call fails, so the calling code always gets a non-None float.
        """
        if self._api is None:
            return float(self.live_balances.get("equity", 0.0))
        try:
            acct = self._api.get_account()
            return float(acct.equity)
        except Exception as exc:
            logger.error(f"[AlpacaBroker] get_account_equity error: {exc}")
            return float(self.live_balances.get("equity", 0.0))

    def compute_total_equity(self) -> float:
        """
        Alias for get_account_equity() — mirrors LiveBroker interface.

        When live_balances is already populated (after sync_account_state)
        this avoids a redundant API call by reading the cached value.
        Upstream code that polls equity frequently should call
        sync_account_state() periodically to refresh the cache.
        """
        cached = float(self.live_balances.get("equity", 0.0))
        if cached > 0:
            return cached
        return self.get_account_equity()

    # ------------------------------------------------------------------
    # Positions — required interface
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        """
        Return all open positions as {alpaca_symbol: qty}.

        Updates self.live_positions and syncs internal ETF / spot dicts.
        """
        if self._api is None:
            return dict(self.live_positions)
        try:
            positions = self._api.list_positions()
            result: dict[str, float] = {p.symbol: float(p.qty) for p in positions}
            self.live_positions = result
            # Sync internal ETF / spot position dicts from Alpaca state
            for internal, alpaca_sym in self.etf_pairs.items():
                self.etf_positions[internal] = result.get(alpaca_sym, 0.0)
            for internal, alpaca_sym in self.crypto_pairs.items():
                self.spot_positions[internal] = result.get(alpaca_sym, 0.0)
            return result
        except Exception as exc:
            logger.error(f"[AlpacaBroker] get_positions error: {exc}")
            return dict(self.live_positions)

    # ------------------------------------------------------------------
    # Price — required interface
    # ------------------------------------------------------------------

    def get_price(self, symbol: str) -> float:
        """
        Return the latest price for *symbol* (internal ticker or Alpaca symbol).

        Returns cached price when the live fetch fails so the caller always
        receives a float (possibly 0.0 if never fetched).
        """
        alpaca_sym = self._resolve_symbol(symbol)
        if alpaca_sym is None:
            return self.live_prices.get(symbol, 0.0)
        return self._fetch_single_price(alpaca_sym, symbol)

    # ------------------------------------------------------------------
    # Symbol resolution helpers
    # ------------------------------------------------------------------

    def _resolve_symbol(self, internal: str) -> Optional[str]:
        """Map an internal ticker to its Alpaca symbol, or None if unknown."""
        s = self._all_pairs.get(internal)
        if s:
            return s
        # Caller may already be using an Alpaca symbol directly
        return internal if internal else None

    def _is_crypto(self, alpaca_sym: str) -> bool:
        """Return True when the Alpaca symbol is a crypto pair (contains '/')."""
        return "/" in alpaca_sym

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def fetch_live_prices(self) -> dict[str, float]:
        """
        Fetch latest prices for all tracked assets (crypto + ETFs) in batch.

        Crypto prices are fetched 24/7.
        ETF prices are only fetched during US market hours (to avoid
        unnecessary calls when markets are closed).

        Updates self.live_prices and self.last_price_timestamp.
        Returns the populated price dict.
        """
        if self._api is None:
            logger.warning("[AlpacaBroker] No API client — prices unavailable.")
            return {}

        result: dict[str, float] = {}

        # --- Crypto prices (available 24/7) ------------------------------
        crypto_syms = list(self.crypto_pairs.values())   # ["BTC/USD", ...]
        if crypto_syms:
            try:
                trades = self._api.get_latest_crypto_trades(crypto_syms)
                for internal, alpaca_sym in self.crypto_pairs.items():
                    t = trades.get(alpaca_sym)
                    if t:
                        result[internal] = float(t.price)
            except Exception as exc:
                logger.error(f"[AlpacaBroker] Batch crypto price fetch error: {exc}")
                # Graceful fallback — try individual fetches
                for internal, alpaca_sym in self.crypto_pairs.items():
                    p = self._fetch_single_price(alpaca_sym, internal)
                    if p > 0:
                        result[internal] = p

        # --- ETF / Stock prices (market hours only) ----------------------
        if self._market_hours.etf_trading_allowed():
            etf_syms = list(set(self.etf_pairs.values()))   # ["QLD", "TQQQ", ...]
            if etf_syms:
                try:
                    trades = self._api.get_latest_trades(etf_syms)
                    for internal, alpaca_sym in self.etf_pairs.items():
                        t = trades.get(alpaca_sym)
                        if t:
                            result[internal] = float(t.price)
                except Exception as exc:
                    logger.error(f"[AlpacaBroker] Batch ETF price fetch error: {exc}")
                    for internal, alpaca_sym in self.etf_pairs.items():
                        p = self._fetch_single_price(alpaca_sym, internal)
                        if p > 0:
                            result[internal] = p

        if result:
            self.live_prices          = result
            self.last_price_timestamp = time.time()
            price_line = "  ".join(f"{a}=${v:,.4f}" for a, v in result.items())
            logger.info(f"[ALPACA PRICE FEED] {price_line}")

            # Warn on any missing assets (crypto expected to always be present)
            missing_crypto = [a for a in self.crypto_pairs if a not in result]
            if missing_crypto:
                logger.warning(f"[AlpacaBroker] Missing crypto prices: {missing_crypto}")

        return result

    def _fetch_single_price(self, alpaca_sym: str, internal: str) -> float:
        """Fetch price for a single Alpaca symbol, routing crypto vs equity."""
        if self._api is None:
            return self.live_prices.get(internal, 0.0)
        try:
            if self._is_crypto(alpaca_sym):
                trade = self._api.get_latest_crypto_trade(alpaca_sym)
            else:
                trade = self._api.get_latest_trade(alpaca_sym)
            return float(trade.price)
        except Exception as exc:
            logger.warning(f"[AlpacaBroker] Single price fetch failed {alpaca_sym}: {exc}")
            return self.live_prices.get(internal, 0.0)

    def prices_are_fresh(self, max_age_sec: float = 10.0) -> bool:
        """Return True when the cached prices are recent enough to act on."""
        if not self.live_prices:
            return False
        return (time.time() - self.last_price_timestamp) <= max_age_sec

    def update_prices_if_needed(self):
        """Refresh prices if stale; trigger kill switch if refresh fails."""
        if not self.prices_are_fresh():
            logger.info("[AlpacaBroker] Refreshing live prices...")
            self.fetch_live_prices()
        if not self.prices_are_fresh():
            self.trigger_kill_switch("Stale or missing live prices — halting order flow")

    # ------------------------------------------------------------------
    # Account sync — required interface
    # ------------------------------------------------------------------

    def fetch_live_balances(self) -> Optional[dict]:
        """
        Fetch account information from Alpaca and populate self.live_balances.

        Returns the balances dict on success, None on failure.
        """
        if self._api is None:
            return None
        try:
            acct = self._api.get_account()
            self.live_balances = {
                "equity":             float(acct.equity),
                "cash":               float(acct.cash),
                "buying_power":       float(acct.buying_power),
                "portfolio_value":    float(acct.portfolio_value),
                "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
                "daytrade_count":     int(getattr(acct, "daytrade_count", 0)),
                "account_type":       str(getattr(acct, "account_type", "MARGIN")),
            }
            return self.live_balances
        except Exception as exc:
            logger.error(f"[AlpacaBroker] fetch_live_balances error: {exc}")
            return None

    def sync_account_state(self) -> Optional[tuple]:
        """
        Fetch balances + positions from Alpaca and run compliance checks.

        Mirrors LiveBroker.sync_live_account_state().
        Returns (balances, positions) on success, None on failure.
        Kill-switch fires if the balance fetch fails.
        """
        if not self.check_health():
            logger.warning("[AlpacaBroker] Kill-switch active — skipping account sync")
            return None

        logger.info("[AlpacaBroker] Syncing account state from Alpaca...")
        balances  = self.fetch_live_balances()
        positions = self.get_positions()

        if balances is None:
            self.trigger_kill_switch("Failed to sync account state from Alpaca")
            return None

        # PDT flag warning
        if balances.get("pattern_day_trader"):
            logger.warning(
                "[AlpacaBroker] Account is flagged as Pattern Day Trader. "
                "Day trading will be blocked when equity < $25,000."
            )

        # Scale safety caps now that real equity is known
        self._sync_safety_caps_to_equity()
        return balances, positions

    # Alias for callers that use LiveBroker naming
    sync_live_account_state = sync_account_state

    # ------------------------------------------------------------------
    # PDT compliance — required interface
    # ------------------------------------------------------------------

    def check_pdt_compliance(self, symbol: str) -> bool:
        """
        Return True if placing a day trade for *symbol* is PDT-safe.

        Pattern Day Trader rule (FINRA Rule 4210):
          A "pattern day trader" is a margin account that executes 4 or more
          day trades (same-day buy-then-sell or sell-then-buy) in a rolling
          5-business-day period.  When flagged, the account must maintain
          ≥ $25,000 in equity or be restricted from placing new day trades.

        Cash accounts are exempt from the PDT rule (though subject to
        T+2 settlement constraints — we log a note rather than block).

        Parameters
        ----------
        symbol : internal ticker (e.g. "ETHD") or Alpaca ticker ("SQQQ")
                 used for contextual logging only.

        Returns
        -------
        True  — order may proceed
        False — order should be blocked to avoid PDT consequences
        """
        balances = self.live_balances
        if not balances:
            # No account data yet — permissive default (check will run next cycle)
            return True

        equity       = float(balances.get("equity",             0.0))
        is_pdt       = bool( balances.get("pattern_day_trader", False))
        daytrade_cnt = int(  balances.get("daytrade_count",     0))
        acct_type    = str(  balances.get("account_type",       "MARGIN")).upper()

        if acct_type == "CASH":
            # Cash accounts: PDT rule does not apply; note for transparency.
            logger.debug("[PDT] Cash account — PDT rule does not apply.")
            return True

        # Margin account: enforce PDT boundaries
        if is_pdt and equity < 25_000:
            logger.warning(
                f"[PDT BLOCK] {symbol}: account flagged as Pattern Day Trader "
                f"with equity ${equity:,.2f} < $25,000 — day trade blocked."
            )
            return False

        if equity < 25_000 and daytrade_cnt >= 3:
            logger.warning(
                f"[PDT BLOCK] {symbol}: {daytrade_cnt} day trades already executed "
                f"in the rolling 5-business-day window "
                f"(equity ${equity:,.2f} < $25,000). "
                "A 4th trade would trigger the PDT flag — blocking order."
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Order placement — required interface
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol:     str,
        qty:        float,
        side:       str,
        order_type: str           = "market",
        price:      Optional[float] = None,
    ) -> Optional[dict]:
        """
        Place a single order on Alpaca.

        Parameters
        ----------
        symbol     : internal ticker (e.g. "BTC", "ETHD") or Alpaca symbol
        qty        : unsigned quantity in base-asset units (coins for crypto,
                     shares for ETFs/stocks)
        side       : "buy" or "sell"
        order_type : "market" or "limit"
        price      : limit price in USD; required when order_type="limit"

        Returns the Alpaca order response dict on success, or None on failure.
        Soft errors (insufficient funds, below minimum) return None without
        triggering the kill switch.  Hard/unexpected errors trigger it.
        """
        if not self.check_health():
            logger.warning(f"[AlpacaBroker] Kill-switch active — order blocked: {symbol}")
            return None

        if side not in ("buy", "sell"):
            logger.error(f"[AlpacaBroker] Invalid side {side!r} for {symbol}")
            return None

        if qty <= 0:
            logger.debug(f"[AlpacaBroker] Zero qty — skipping {symbol}")
            return None

        alpaca_sym = self._resolve_symbol(symbol)
        if alpaca_sym is None:
            logger.error(f"[AlpacaBroker] Unknown symbol: {symbol!r}")
            return None

        is_crypto = self._is_crypto(alpaca_sym)

        # PDT compliance check for equity / ETF orders only (crypto is 24/7 spot)
        if not is_crypto and not self.check_pdt_compliance(symbol):
            return None

        # Crypto uses IOC (Immediate-Or-Cancel); stocks/ETFs use day orders
        tif = "ioc" if is_crypto else "day"

        order_params: dict = {
            "symbol":        alpaca_sym,
            "qty":           qty,
            "side":          side,
            "type":          order_type,
            "time_in_force": tif,
        }
        if order_type == "limit":
            if price is None or price <= 0:
                logger.warning(
                    f"[AlpacaBroker] Limit order for {symbol} missing price — "
                    "downgrading to market order."
                )
                order_params["type"] = "market"
            else:
                order_params["limit_price"] = str(round(price, 6))

        if self.dry_run:
            logger.info(
                f"[ALPACA DRY-RUN] {side.upper()} {qty} {alpaca_sym}"
                f"  type={order_params['type']}"
                + (f"  limit=${price:.4f}" if price else "")
            )
            return {
                "status": "dry_run",
                "symbol": alpaca_sym,
                "side":   side,
                "qty":    qty,
                "type":   order_params["type"],
            }

        if self._api is None:
            logger.error("[AlpacaBroker] No API client — cannot place order.")
            return None

        try:
            order = self._api.submit_order(**order_params)
            logger.info(
                f"[ALPACA ORDER] id={order.id}  {side.upper()} {qty} {alpaca_sym}"
                f"  status={order.status}"
            )
            # Return a plain dict so callers don't need to depend on alpaca types
            return {
                "id":     order.id,
                "status": order.status,
                "symbol": alpaca_sym,
                "side":   side,
                "qty":    qty,
            }
        except Exception as exc:
            err = str(exc)
            soft_keywords = (
                "insufficient", "minimum", "too small", "notional",
                "order minimum not met", "buying power",
            )
            if any(kw in err.lower() for kw in soft_keywords):
                logger.warning(f"[ALPACA ORDER SKIPPED] {symbol}: {err}")
                return None
            # Unexpected / hard error — trigger kill switch
            self.trigger_kill_switch(f"Order submission failed for {symbol}: {err}")
            return None

    # ------------------------------------------------------------------
    # Spot trade execution (mirrors LiveBroker._execute_spot_trade)
    # ------------------------------------------------------------------

    def _execute_spot_trade(
        self,
        asset:           str,
        price:           float,
        delta_exposure:  float,
        microstructure_fn=None,
    ) -> float:
        """
        Execute a crypto spot trade on Alpaca.

        delta_exposure is a *fractional* delta of total equity (e.g. +0.06
        = buy 6% of portfolio value in this asset).

        Returns the estimated fee in USD so the caller can deduct it from
        tracked balance (mirrors LiveBroker._execute_spot_trade return value).
        """
        equity       = self.compute_total_equity() or 1.0
        notional_usd = abs(delta_exposure) * equity
        side         = "buy" if delta_exposure > 0 else "sell"

        if not self._pre_trade_safety(asset, price, notional_usd):
            logger.debug(f"[AlpacaBroker] Pre-trade safety blocked {asset}")
            return 0.0

        if price <= 0:
            return 0.0

        qty = round(notional_usd / price, 8)
        if qty <= 0:
            return 0.0

        fee_usd = notional_usd * self.taker_fee

        result = self.place_order(
            symbol     = asset,
            qty        = qty,
            side       = side,
            order_type = "market",
        )

        if result is None:
            return 0.0

        # Update internal position tracking
        delta = qty if side == "buy" else -qty
        self.spot_positions[asset] = self.spot_positions.get(asset, 0.0) + delta
        self.cumulative_fees_usd  += fee_usd

        logger.info(
            f"[ALPACA SPOT] {side.upper()} {qty:.6f} {asset}"
            f"  @ ${price:.4f}  notional=${notional_usd:.2f}  fee≈${fee_usd:.4f}"
        )

        if not self.dry_run:
            self._log_live_trade(asset, side, qty, price, notional_usd, fee_usd)

        return fee_usd

    # ------------------------------------------------------------------
    # ETF overlay (mirrors LiveBroker.run_etf_overlay)
    # ------------------------------------------------------------------

    def run_etf_overlay(
        self,
        agent,
        prices: dict,
        regime: Optional[dict] = None,
    ):
        """
        ETF hedging / amplification pass using Alpaca equity ETFs.

        Computes target ETF exposures via the exchange-agnostic ETFHedger,
        then routes orders through Alpaca's stock/ETF API using the
        ALPACA_ETF_PAIRS symbol mapping:

            ETHD → SQQQ  (primary bear-market hedge)
            SETH → SH    (lighter bear-market hedge)
            ETHU → QLD   (bull-market amplifier)
            SLON → TQQQ  (bull-market amplifier)
            XXRP → SSO   (bull-market amplifier)

        Behaviour mirrors LiveBroker.run_etf_overlay() exactly:
          • Market orders during regular session (09:30–16:00 ET).
          • Limit orders during pre/after-market.
          • No ETF orders when market is closed (weekends, overnight).
          • Combined allocation capped at MAX_ETF_ALLOCATION (default 30%).
          • Profit-maximising exit gate: positions are held through neutral
            signals to avoid unnecessary fee drag.

        Parameters
        ----------
        agent  : (ignored) kept for API compatibility with LiveBroker callers
        prices : dict {internal_ticker: float} — live prices
        regime : optional regime dict (cycle_phase, panic_risk, …)
        """
        if not self.check_health():
            return

        if not self._check_daily_loss():
            return

        if not self._market_hours.etf_trading_allowed():
            logger.info(f"[ETF OVERLAY] {self._market_hours.status_line()} — skipping")
            return

        regime = regime or {}
        equity = self.compute_total_equity()
        if equity <= 0:
            return

        # Build ETF price dict from live_prices or the provided prices arg
        etf_prices: dict[str, float] = {}
        for a in ETF_ASSETS:
            p = self.live_prices.get(a) or prices.get(a, 0.0)
            if p > 0:
                etf_prices[a] = p

        if not etf_prices:
            logger.warning("[ETF OVERLAY] No ETF prices available — skipping")
            return

        # Enforce combined 30% cap before computing new orders
        if self.etf_hedger.cap_breached(equity, self.etf_positions, etf_prices):
            frac = self.etf_hedger.etf_portfolio_fraction(
                equity, self.etf_positions, etf_prices
            )
            logger.info(
                f"[ETF OVERLAY] 30% cap already breached (frac={frac:.3f}) — skipping"
            )
            return

        orders = self.etf_hedger.compute_orders(
            regime        = regime,
            equity        = equity,
            etf_prices    = etf_prices,
            etf_positions = self.etf_positions,
        )

        _curr_dir  = _etf_regime_direction(regime)
        _prev_dir  = self._prev_etf_regime_dir
        _taker_fee = self.taker_fee

        if not orders:
            logger.debug(
                f"[ETF OVERLAY] No rebalance orders needed"
                f"  regime_dir={_curr_dir}  (was {_prev_dir})"
            )
            if _curr_dir != "neutral":
                self._prev_etf_regime_dir = _curr_dir
            return

        order_type = self._market_hours.required_order_type()
        limit_tol  = float(os.getenv("LIMIT_ORDER_TOLERANCE", "0.001"))

        for order in orders:
            asset    = order["asset"]      # internal ticker (e.g. "ETHD")
            side     = order["side"]
            units    = order["units"]
            notional = order["notional"]

            # ---- Profit-maximising exit gate (sell orders only) ---------
            if side == "sell":
                if _curr_dir == "neutral":
                    # Signal is indeterminate — retain the position.
                    current_usd = (
                        self.etf_positions.get(asset, 0.0)
                        * etf_prices.get(asset, 0.0)
                    )
                    logger.info(
                        f"  [ETF OVERLAY] HOLD {asset}"
                        f"  position=${current_usd:.2f}"
                        f"  — signal neutral (was {_prev_dir})"
                        f"  — retaining for profit maximisation"
                    )
                    continue

                round_trip_cost = notional * 2.0 * _taker_fee
                fee_hurdle      = round_trip_cost * _ETF_FEE_HURDLE_FACTOR
                if notional < fee_hurdle:
                    logger.info(
                        f"  [ETF OVERLAY] HOLD {asset}"
                        f"  — exit gain ${notional:.2f} < fee hurdle ${fee_hurdle:.2f}"
                        f"  — holding for profit maximisation"
                    )
                    continue
            # ---- End exit gate ------------------------------------------

            # Map internal ticker to Alpaca stock ticker
            alpaca_sym = self.etf_pairs.get(asset)
            if alpaca_sym is None:
                logger.warning(f"[ETF OVERLAY] No Alpaca mapping for internal ticker {asset!r}")
                continue

            etf_price = etf_prices.get(asset, 0.0)
            if not self._pre_trade_safety(asset, etf_price, notional):
                logger.debug(
                    f"  [ETF OVERLAY] Pre-trade safety blocked {side.upper()} {asset}"
                )
                continue

            # Determine limit price for extended-hours sessions
            lim_price: Optional[float] = None
            if order_type == "limit" and etf_price > 0:
                lim_price = (
                    etf_price * (1.0 + limit_tol)
                    if side == "buy"
                    else etf_price * (1.0 - limit_tol)
                )

            result = self.place_order(
                symbol     = asset,
                qty        = round(units, 4),
                side       = side,
                order_type = order_type,
                price      = lim_price,
            )

            if result is not None:
                delta = units if side == "buy" else -units
                self.etf_positions[asset] = self.etf_positions.get(asset, 0.0) + delta
                # ETF stock trades are commission-free on Alpaca
                fee_usd = 0.0
                self.cumulative_fees_usd += fee_usd
                logger.info(
                    f"  [ETF OVERLAY] {side.upper()} {units:.4f} {asset}"
                    f" → {alpaca_sym}  notional=${notional:.2f}"
                )
                if not self.dry_run:
                    self._log_live_trade(
                        asset, side, units, etf_price, notional, fee_usd
                    )

        if _curr_dir != "neutral":
            self._prev_etf_regime_dir = _curr_dir

    # ------------------------------------------------------------------
    # Credential validation — required interface
    # ------------------------------------------------------------------

    def validate_credentials(self) -> bool:
        """
        Confirm that ALPACA_API_KEY and ALPACA_API_SECRET are valid by
        calling the Alpaca GET /v2/account endpoint.

        Returns True on success, False on any failure.  Does NOT trigger
        the kill switch — the caller decides whether to abort or fall back
        to dry-run mode.
        """
        if self._api is None:
            logger.error("[AlpacaBroker] validate_credentials: no API client.")
            return False
        try:
            acct   = self._api.get_account()
            equity = float(acct.equity)
            logger.info(
                f"[AlpacaBroker] ✅ Credentials valid — account equity: ${equity:,.2f}"
            )
            return True
        except Exception as exc:
            logger.error(f"[AlpacaBroker] validate_credentials failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Market state helpers (for compatibility with callers)
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True when the US equity market is currently open."""
        if self._api is None:
            return self._market_hours.etf_trading_allowed()
        try:
            clock = self._api.get_clock()
            return bool(clock.is_open)
        except Exception:
            return self._market_hours.etf_trading_allowed()

    # ------------------------------------------------------------------
    # Safety checks (mirrors LiveBroker)
    # ------------------------------------------------------------------

    def _pre_trade_safety(
        self,
        asset:        str,
        price:        float,
        notional_usd: float,
    ) -> bool:
        """
        Unified pre-trade gate.  Returns False (and logs why) if any check fails.

        Checks (in order):
          1. Kill switch
          2. Valid price
          3. Minimum notional
          4. Per-asset notional cap
          5. Trade rate limit
          6. Daily loss cap
        """
        if not self.check_health():
            return False

        if price is None or price <= 0:
            logger.warning(f"[AlpacaBroker] Invalid price for {asset}: {price!r}")
            return False

        if notional_usd < _MIN_NOTIONAL_USD:
            logger.debug(
                f"[AlpacaBroker] Below min notional: {asset} ${notional_usd:.2f} "
                f"< ${_MIN_NOTIONAL_USD:.2f}"
            )
            return False

        if notional_usd > self.max_notional_per_asset:
            logger.warning(
                f"[AlpacaBroker] Per-asset notional cap: {asset}"
                f"  ${notional_usd:.2f} > ${self.max_notional_per_asset:.2f}"
            )
            return False

        if not self._check_trade_rate():
            return False

        if not self._check_daily_loss():
            return False

        return True

    def _check_trade_rate(self) -> bool:
        """Enforce a rolling 1-hour trade rate limit."""
        now    = time.time()
        cutoff = now - 3600
        self._trade_count_window = [t for t in self._trade_count_window if t > cutoff]
        if len(self._trade_count_window) >= self.max_trades_per_hour:
            self.trigger_kill_switch("Trade rate exceeded — too many trades in 1 h")
            return False
        self._trade_count_window.append(now)
        return True

    def _check_daily_loss(self) -> bool:
        """
        Compare current equity to session-start equity.
        Triggers kill switch if loss exceeds max_daily_loss_pct.
        """
        equity = self.compute_total_equity()

        if self._starting_equity is None:
            if equity > 0:
                self._starting_equity    = equity
                self._max_daily_loss_usd = round(equity * self.max_daily_loss_pct, 2)
                logger.info(
                    f"[AlpacaBroker] Session equity anchored at ${equity:.2f}  "
                    f"→ daily loss cap = ${self._max_daily_loss_usd:.2f}"
                    f" ({self.max_daily_loss_pct:.0%})"
                )
                self._sync_safety_caps_to_equity()
            return True

        if equity <= 0:
            return True   # no balance data — permissive

        loss_cap = (
            self._max_daily_loss_usd
            or (self._starting_equity * self.max_daily_loss_pct)
        )
        if equity < self._starting_equity - loss_cap:
            self.trigger_kill_switch(
                f"Daily loss limit exceeded: equity=${equity:.2f}  "
                f"start=${self._starting_equity:.2f}  cap=${loss_cap:.2f}"
                f" ({self.max_daily_loss_pct:.0%})"
            )
            return False

        return True

    def _sync_safety_caps_to_equity(self) -> None:
        """
        Scale per-asset and total notional caps to current account equity.

        Prevents the conservative first-session defaults ($50 per asset,
        $200 total) from blocking legitimate orders on larger accounts.
        """
        equity = self.compute_total_equity()
        if equity <= 0:
            return
        if not self._per_asset_cap_fixed:
            self.max_notional_per_asset = equity * self._max_notional_pct
        if not self._total_cap_fixed:
            self.max_total_notional = equity * self._max_total_notional_pct
        logger.debug(
            f"[AlpacaBroker] Notional caps: per_asset=${self.max_notional_per_asset:.2f}"
            f"  total=${self.max_total_notional:.2f}"
        )

    # ------------------------------------------------------------------
    # Health / heartbeat (mirrors LiveBroker — reduced implementation)
    # ------------------------------------------------------------------

    def heartbeat(self, interval_seconds: int = 300):
        """Emit a health log line at most once per interval_seconds."""
        now = time.time()
        if not hasattr(self, "_last_heartbeat"):
            self._last_heartbeat = 0.0
        if now - self._last_heartbeat >= interval_seconds:
            if self.kill_switch:
                logger.warning("[AlpacaBroker HEARTBEAT] Kill switch active")
            else:
                logger.info(
                    f"[AlpacaBroker HEARTBEAT] alive  "
                    f"equity=${self.compute_total_equity():.2f}  "
                    f"fees=${self.cumulative_fees_usd:.2f}"
                )
            self._last_heartbeat = now

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _log_live_trade(
        self,
        asset:      str,
        side:       str,
        qty:        float,
        fill_price: float,
        notional:   float,
        fee:        float,
    ):
        """Append a filled-trade record to the CSV audit log."""
        try:
            self._live_csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                asset, side,
                round(qty,        8),
                round(fill_price, 6),
                round(notional,   4),
                round(fee,        6),
            ])
            self._live_csv_file.flush()
        except Exception as exc:
            logger.error(f"[AlpacaBroker] CSV log error: {exc}")

    def __del__(self):
        """Close the audit log file on garbage collection."""
        try:
            if hasattr(self, "_live_csv_file") and self._live_csv_file:
                self._live_csv_file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Broker factory — called by the main runner when BROKER=alpaca is set
# ---------------------------------------------------------------------------

def create_alpaca_broker(dry_run: bool = True) -> AlpacaBroker:
    """
    Convenience factory that derives the effective dry_run mode from both
    the caller's preference and the ENABLE_LIVE_TRADING environment variable.

    Live order submission requires BOTH conditions to be true:
      1. dry_run=False is explicitly passed by the caller, AND
      2. ENABLE_LIVE_TRADING=true is set in the environment.

    If either condition is not met the broker defaults to dry_run=True
    (no real orders), which is always the safe fallback.

    Typical usage in the main runner
    ----------------------------------
      # Paper mode (default):
      broker = create_alpaca_broker()

      # Live mode (set ENABLE_LIVE_TRADING=true in .env first):
      broker = create_alpaca_broker(dry_run=False)
    """
    _live_env = os.getenv("ENABLE_LIVE_TRADING", "false").lower()
    _env_live = _live_env not in ("false", "0", "no")
    # Live trading requires BOTH the caller to set dry_run=False AND the
    # environment variable ENABLE_LIVE_TRADING=true.  Any other combination
    # keeps dry_run=True for safety.
    effective_dry_run = not (_env_live and not dry_run)
    return AlpacaBroker(dry_run=effective_dry_run)
