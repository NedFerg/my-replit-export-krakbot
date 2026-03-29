"""
project/broker/etf_hedging.py
------------------------------
ETF hedging strategy and order builder.

This module replaces the disabled futures hedging layer for US users by
using Kraken Spot positions in crypto ETPs (bare ticker symbols, no suffix):

  Long ETPs:
    ETHU  — ETH 2× Long ETP
    SLON  — SOL 2× Long ETP
    XXRP  — XRP 2× Long ETP

  Short ETPs (only these two are short):
    ETHD  — ETH 2× Short ETP
    SETH  — ETH 1× Short ETP

Because Ethereum leads the altcoin market, shorting SETH/ETHD acts as a broad
crypto hedge during bearish trends — the same role futures hedging played.

Design
------
• ETF positions are stored separately in broker.etf_positions (never mixed
  with spot crypto positions in broker.spot_positions).
• Combined |ETHD| + |SETH| notional ≤ 30 % of total equity at all times.
• Order type (market vs. limit) is enforced by MarketHours:
    – Regular hours (09:30–16:00 ET): market orders
    – Pre/after-market (04:00–09:30 and 16:00–20:00 ET): limit orders only
    – Closed (overnight / weekends): no ETF orders

The module is intentionally thin: it computes *what* to do and returns an
order dict.  The broker decides *whether* to submit it (kill-switch checks,
daily-loss caps, etc. still apply in the broker layer).

Usage (called from broker.run_etf_overlay())
-----
    from broker.etf_hedging import ETFHedger

    hedger = ETFHedger(max_etf_allocation=0.30)
    orders = hedger.compute_orders(
        regime       = {"cycle_phase": 3, "panic_risk": 2},
        equity       = 10_000.0,
        etf_prices   = {"ETHD": 25.0, "SETH": 8.0},
        etf_positions= {"ETHD": 0.0, "SETH": 0.0},
    )
    # orders is a list of dicts with keys: asset, side, units, order_type, price
"""

import enum
import os
from typing import NamedTuple

from utils.market_hours import MarketHours, MarketSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ETF_ASSETS = ("ETHU", "SLON", "XXRP", "ETHD", "SETH")

# Supported ETFs for the regime-based ETF overlay logic.
# ETHU and XXRP are 2x long ETFs; ETHD and SETH are approved short (inverse) ETFs.
ALL_ETFS = ["ETHU", "XXRP", "ETHD", "SETH"]

# Default Kraken trading pairs for each ETF ticker.
# ETFs/ETPs on Kraken use bare ticker symbols (no currency suffix).
ETF_KRAKEN_PAIRS: dict = {
    "ETHU": "ETHU",      # ETH 2× Long ETP
    "SLON": "SLON",      # SOL 2× Long ETP
    "XXRP": "XXRP",      # XRP 2× Long ETP
    "ETHD": "ETHD",      # ETH 2× Short ETP
    "SETH": "SETH",      # ETH 1× Short ETP
}


class ETFMode(str, enum.Enum):
    """Operating mode for the ETF hedging layer."""
    DISABLED = "disabled"
    HEDGE    = "hedge"
    AMPLIFY  = "amplify"


class ETFOrder(NamedTuple):
    """
    Typed container for a single ETF order produced by ETFHedger.

    Fields
    ------
    asset      : ticker symbol, e.g. "ETHD" or "SETH"
    side       : "buy" or "sell"
    units      : unsigned coin quantity to trade
    order_type : "market" or "limit"
    price      : limit price in the quote currency; 0.0 for market orders
    notional   : estimated order value in the quote currency (|units × price|)
    """
    asset:      str
    side:       str
    units:      float
    order_type: str
    price:      float
    notional:   float

# ---------------------------------------------------------------------------
# Bull Market Mode — reads the shared BULL_MARKET_MODE env var
# ---------------------------------------------------------------------------
# When enabled, the ETF allocation cap is raised from the default 30% to 50%.
# The env var MAX_ETF_ALLOCATION always takes explicit precedence over both
# the bull-mode default and the standard default.
_BULL_MARKET_MODE: bool = os.getenv("BULL_MARKET_MODE", "false").lower() in ("1", "true", "yes")

# Default allocation cap — overridden by MAX_ETF_ALLOCATION env var or the
# max_etf_allocation constructor parameter.
# Bull mode default: 0.50 (50%).  Standard default: 0.30 (30%).
_DEFAULT_ETF_CAP: float = 0.50 if _BULL_MARKET_MODE else 0.30
DEFAULT_MAX_ETF_ALLOCATION = float(os.getenv("MAX_ETF_ALLOCATION", str(_DEFAULT_ETF_CAP)))

# Minimum notional USD per order — skip dust trades
MIN_ORDER_USD = float(os.getenv("ETF_MIN_ORDER_USD", "5.0"))

# Minimum USD threshold for priority ETF allocation (30% of available cash
# must exceed this value before attempting an ETF order).  Configurable via
# ETF_MIN_ALLOCATION_USD env var; default $20 avoids rejected min-notional
# orders and prevents stalling on trivially small accounts.
ETF_MIN_ALLOCATION_USD = float(os.getenv("ETF_MIN_ALLOCATION_USD", "20.0"))

# Timeout (seconds) after which a pending/unfilled ETF priority order is
# considered stale and the lock is cleared so spot trading can proceed.
# Default 30 minutes (1800 s); configurable via ETF_ORDER_TIMEOUT_SEC.
ETF_ORDER_TIMEOUT_SEC = int(os.getenv("ETF_ORDER_TIMEOUT_SEC", "1800"))

# Limit order tolerance: how far from mid the limit price is placed.
# 0.1 % matches the spot limit_order_tolerance default in broker.py.
DEFAULT_LIMIT_TOLERANCE = float(os.getenv("ETF_LIMIT_TOLERANCE", "0.001"))


# ---------------------------------------------------------------------------
# Regime direction classifier
# ---------------------------------------------------------------------------

def etf_regime_direction(regime: dict) -> str:
    """
    Classify the current market regime as 'bull', 'bear', or 'neutral'.

    Used by the ETF overlay to distinguish between:
      • A **clear reversal** (bull→bear or bear→bull) — where exiting an
        existing position and rotating to the opposite ETF is justified if
        the expected gain clears the round-trip fee hurdle.
      • A **neutral / indeterminate signal** — where the regime has not
        yet committed to a new direction.  In this case existing ETF
        positions are held rather than closed, because closing and
        re-entering later incurs two sets of fees and risks missing the
        recovery move (profit-maximizing hold).

    Inputs (all optional — missing keys default to neutral):
      panic_risk          int {0,1,2}  — 0 = calm, 1 = elevated, 2 = panic
      macro_regime        float        — +1 = bull, 0 = neutral, -1 = bear
      bullish_confidence  float [0,1]  — mean positive agent exposure
      bearish_drift       bool         — True when ETH 20-bar momentum < -1 %
      cycle_phase         int {0-3}    — 0/1 = bull phases, 2/3 = bear phases

    Returns
    -------
    str — one of "bull", "bear", "neutral"
    """
    panic       = int(regime.get("panic_risk", 0))
    macro       = float(regime.get("macro_regime", 0.0))
    confidence  = float(regime.get("bullish_confidence", 0.0))
    bearish     = bool(regime.get("bearish_drift", False))
    cycle_phase = int(regime.get("cycle_phase", -1))

    # Bear signals take priority (protect capital first)
    if panic >= 1 or macro <= -0.5 or bearish:
        return "bear"

    # Bull signals: strong macro momentum OR high agent confidence OR
    # accumulation/expansion cycle phase
    if macro >= 0.5 or confidence >= 0.5 or cycle_phase in (0, 1):
        return "bull"

    return "neutral"


# ---------------------------------------------------------------------------
# Regime → target-exposure mapping
# ---------------------------------------------------------------------------

def _regime_to_etf_targets(regime: dict) -> dict:
    """
    Map a market regime snapshot to target fractional ETF exposures.

    Inputs (all optional — missing keys default to neutral):
      cycle_phase  int 0-3   (0=accumulation, 1=expansion, 2=distribution, 3=reset)
      panic_risk   int 0-2   (0=none, 1=moderate, 2=severe)
      top_risk     int 0-2   (0=none, 1=moderate, 2=severe)
      vol_scaler   float     (>1 = elevated volatility)

    Target exposure conventions:
      ETHD  ≥ 0  (long exposure; 0 = no position)
      SETH  ≥ 0  (short exposure expressed as a positive allocation fraction;
                  the order side is always "buy" when opening, "sell" when closing)

    Returns dict {"ETHD": float, "SETH": float} — each in [0, 1].
    The caller enforces the 30 % portfolio cap after combining both.
    """
    cycle_phase = int(regime.get("cycle_phase", 1))
    panic_risk  = int(regime.get("panic_risk",  0))
    top_risk    = int(regime.get("top_risk",    0))
    vol_scaler  = float(regime.get("vol_scaler", 1.0))

    # Severe panic → max short hedge; minimal or no long
    if panic_risk == 2:
        return {"ETHD": 0.00, "SETH": 0.30}

    # Severe blow-off top → reduce long, open moderate short
    if top_risk == 2:
        return {"ETHD": 0.00, "SETH": 0.20}

    # Moderate panic → partial short, no long
    if panic_risk == 1:
        return {"ETHD": 0.00, "SETH": 0.15}

    # Moderate top → trim long, small short
    if top_risk == 1:
        return {"ETHD": 0.05, "SETH": 0.10}

    # Cycle-phase base targets (no extreme signals)
    phase_map = {
        0: {"ETHD": 0.10, "SETH": 0.00},   # accumulation: small long hedge
        1: {"ETHD": 0.15, "SETH": 0.00},   # expansion: moderate long hedge
        2: {"ETHD": 0.05, "SETH": 0.05},   # distribution: balanced
        3: {"ETHD": 0.00, "SETH": 0.20},   # reset/crash: short hedge
    }
    targets = phase_map.get(cycle_phase, {"ETHD": 0.0, "SETH": 0.0})

    # Scale down slightly when volatility is very elevated
    if vol_scaler > 1.5:
        scale = max(0.5, 1.0 / vol_scaler)
        targets = {k: v * scale for k, v in targets.items()}

    return targets


# ---------------------------------------------------------------------------
# Order builder
# ---------------------------------------------------------------------------

class ETFHedger:
    """
    Stateless ETF hedging strategy.

    All state (current positions, prices) is passed in by the caller so this
    object has no internal mutable state and is safe to call from any thread.
    """

    def __init__(
        self,
        max_etf_allocation: float = DEFAULT_MAX_ETF_ALLOCATION,
        limit_tolerance: float    = DEFAULT_LIMIT_TOLERANCE,
    ):
        self.max_etf_allocation = max_etf_allocation
        self.limit_tolerance    = limit_tolerance
        self._market_hours      = MarketHours()
        if _BULL_MARKET_MODE:
            print(
                f"[ETFHedger] ⚠️  Bull Market Mode enabled: ETF allocation cap raised "
                f"to {self.max_etf_allocation*100:.0f}% (standard: 30%). "
                "See README.md for details."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_orders(
        self,
        regime:        dict,
        equity:        float,
        etf_prices:    dict,
        etf_positions: dict,
        now=None,
    ) -> list:
        """
        Compute ETF rebalance orders for the current regime and positions.

        Parameters
        ----------
        regime        : dict with keys cycle_phase, panic_risk, top_risk, vol_scaler
        equity        : total portfolio equity in USD
        etf_prices    : {"ETHD": float, "SETH": float} — current mid-prices
        etf_positions : {"ETHD": float, "SETH": float} — current coin holdings
        now           : optional datetime for market-hours determination (for tests)

        Returns
        -------
        list of order dicts, each with:
            {
                "asset":      str,    # "ETHD" or "SETH"
                "side":       str,    # "buy" or "sell"
                "units":      float,  # unsigned coin quantity
                "order_type": str,    # "market" or "limit"
                "price":      float,  # limit price (0.0 when market order)
                "notional":   float,  # estimated USD notional
            }

        An empty list means no action is required.
        """
        if equity <= 0:
            return []

        if not self._market_hours.etf_trading_allowed(now):
            return []

        session    = self._market_hours.get_session(now)
        order_type = self._market_hours.required_order_type(now)

        targets = _regime_to_etf_targets(regime)

        # Enforce combined 30 % cap
        total_target = sum(targets.values())
        if total_target > self.max_etf_allocation:
            scale   = self.max_etf_allocation / total_target
            targets = {k: v * scale for k, v in targets.items()}

        orders = []
        for asset in ETF_ASSETS:
            price = etf_prices.get(asset, 0.0)
            if price <= 0:
                continue

            target_frac   = targets.get(asset, 0.0)
            target_usd    = target_frac * equity
            target_units  = target_usd / price

            current_units = float(etf_positions.get(asset, 0.0))
            current_usd   = current_units * price

            delta_usd   = target_usd - current_usd
            delta_units = abs(delta_usd) / price

            if abs(delta_usd) < MIN_ORDER_USD:
                continue

            side = "buy" if delta_usd > 0 else "sell"

            if order_type == "limit":
                if side == "buy":
                    limit_price = round(price * (1.0 + self.limit_tolerance), 8)
                else:
                    limit_price = round(price * (1.0 - self.limit_tolerance), 8)
            else:
                limit_price = 0.0   # market order — no explicit price needed

            orders.append({
                "asset":      asset,
                "side":       side,
                "units":      round(delta_units, 8),
                "order_type": order_type,
                "price":      limit_price,
                "notional":   round(abs(delta_usd), 4),
            })

            print(
                f"[ETF HEDGER] {asset}  session={session}"
                f"  target={target_frac:.3f}  current={current_usd/equity:.3f}"
                f"  delta_usd={delta_usd:+.2f}  side={side}"
                f"  order_type={order_type}"
                + (f"  limit_price={limit_price:.4f}" if order_type == "limit" else "")
            )

        return orders

    def etf_portfolio_fraction(
        self,
        equity:        float,
        etf_positions: dict,
        etf_prices:    dict,
    ) -> float:
        """
        Return the current combined ETF allocation as a fraction of equity.

        sum(|ETHD_usd| + |SETH_usd|) / equity
        """
        if equity <= 0:
            return 0.0
        total = sum(
            abs(etf_positions.get(a, 0.0)) * etf_prices.get(a, 0.0)
            for a in ETF_ASSETS
        )
        return total / equity

    def cap_breached(
        self,
        equity:        float,
        etf_positions: dict,
        etf_prices:    dict,
    ) -> bool:
        """Return True if the current ETF allocation exceeds the 30 % cap."""
        return self.etf_portfolio_fraction(equity, etf_positions, etf_prices) > self.max_etf_allocation


# ---------------------------------------------------------------------------
# Priority ETF selector — used by broker.run_etf_priority_allocation()
# ---------------------------------------------------------------------------

def select_priority_etf(regime: dict) -> tuple:
    """
    Select the single best leveraged ETF for immediate priority deployment.

    This is used when fresh cash is detected (startup or deposit) to
    immediately allocate up to 30 % of available funds to one ETF before
    any spot crypto trading begins.

    Parameters
    ----------
    regime : dict with optional keys: cycle_phase (int 0-3), panic_risk
             (int 0-2), top_risk (int 0-2), vol_scaler (float).

    Returns
    -------
    (asset, target_fraction) : str × float
        asset            — one of ETF_ASSETS
        target_fraction  — fraction of available cash to deploy (0–0.30)

    Selection logic (highest priority first)
    ----------------------------------------
    Severe panic (panic_risk=2)  → SETH  30 % short  (broad crypto crash hedge)
    Severe top   (top_risk=2)    → SETH  20 % short  (blow-off top protection)
    Moderate panic (panic_risk=1)→ SETH  15 % short
    Moderate top  (top_risk=1)   → SETH  15 % short
    Cycle phase 3 (bear/reset)   → SETH  25 % short
    Cycle phase 2 (distribution) → SETH  10 % short  (trend turning)
    Cycle phase 0 (accumulation) → ETHU  10 % long   (early bull)
    Cycle phase 1 (expansion)    → ETHU  20 % long   (confirmed bull)
    Default (neutral / unknown)  → ETHU  15 % long
    """
    cycle_phase = int(regime.get("cycle_phase", 1))
    panic_risk  = int(regime.get("panic_risk",  0))
    top_risk    = int(regime.get("top_risk",    0))

    # Bearish / risk-off signals take highest priority
    if panic_risk == 2:
        return ("SETH", 0.30)
    if top_risk == 2:
        return ("SETH", 0.20)
    if panic_risk == 1:
        return ("SETH", 0.15)
    if top_risk == 1:
        return ("SETH", 0.15)

    # Cycle-phase base selections (no extreme risk signals)
    phase_map = {
        0: ("ETHU", 0.10),   # accumulation — early bull, small long
        1: ("ETHU", 0.20),   # expansion   — confirmed bull, moderate long
        2: ("SETH", 0.10),   # distribution — trend turning, light short
        3: ("SETH", 0.25),   # reset/crash  — bear market, heavy short
    }
    return phase_map.get(cycle_phase, ("ETHU", 0.15))


# ---------------------------------------------------------------------------
# Alias — broker.py imports ETFHedgingLayer; ETFHedger is the implementation.
# ---------------------------------------------------------------------------

ETFHedgingLayer = ETFHedger
