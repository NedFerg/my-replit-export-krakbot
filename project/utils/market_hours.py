"""
project/utils/market_hours.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
US equity / ETF market hours detection with ET timezone support.

Kraken Spot ETFs (SETH, ETHD, XXRP, SLON, ETHU) trade during US market
hours only.  This module classifies the current time into one of three
periods and returns the appropriate Kraken order type:

    Premarket    (04:00–09:30 ET)  → limit orders only
    Regular      (09:30–16:00 ET)  → market orders allowed
    After-hours  (16:00–20:00 ET)  → limit orders only
    Closed       (20:00–04:00 ET)  → no ETF trading

Environment variables (all optional – defaults shown):
    TRADING_TIMEZONE   America/New_York
    MARKET_HOURS_START 09:30
    MARKET_HOURS_END   16:00
    PREMARKET_START    04:00
    AFTERHOURS_END     20:00
"""

from __future__ import annotations

import os
from datetime import datetime, time as dt_time
from enum import Enum
from zoneinfo import ZoneInfo          # stdlib ≥ 3.9


# ---------------------------------------------------------------------------
# Market period enumeration
# ---------------------------------------------------------------------------

class MarketPeriod(Enum):
    CLOSED      = "closed"
    PREMARKET   = "premarket"
    REGULAR     = "regular"
    AFTER_HOURS = "after_hours"


# ---------------------------------------------------------------------------
# Order-type constant returned to callers
# ---------------------------------------------------------------------------

ORDER_TYPE_MARKET = "mkt"
ORDER_TYPE_LIMIT  = "limit"


# ---------------------------------------------------------------------------
# Time boundary helpers
# ---------------------------------------------------------------------------

def _parse_hhmm(value: str, default: dt_time) -> dt_time:
    """Parse 'HH:MM' string to datetime.time, returning *default* on error."""
    try:
        parts = value.strip().split(":")
        return dt_time(int(parts[0]), int(parts[1]))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_market_period(now: datetime | None = None) -> MarketPeriod:
    """
    Classify *now* into a MarketPeriod.

    Parameters
    ----------
    now : datetime | None
        UTC-aware datetime to evaluate.  Defaults to the current UTC time
        when None is passed — making the function zero-argument in production.

    Returns
    -------
    MarketPeriod enum value.

    Notes
    -----
    - Weekends (Saturday, Sunday) always return MarketPeriod.CLOSED.
    - Time boundaries are configurable via environment variables so the same
      module works for premarket-only or extended-hours testing without a
      code change.
    """
    tz_name = os.getenv("TRADING_TIMEZONE", "America/New_York")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    if now is None:
        now = datetime.now(ZoneInfo("UTC"))

    # Convert to ET for comparison
    local = now.astimezone(tz)

    # Weekends: markets closed
    if local.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
        return MarketPeriod.CLOSED

    current = local.time().replace(second=0, microsecond=0)

    premarket_start = _parse_hhmm(os.getenv("PREMARKET_START",    "04:00"), dt_time(4,  0))
    market_open     = _parse_hhmm(os.getenv("MARKET_HOURS_START", "09:30"), dt_time(9,  30))
    market_close    = _parse_hhmm(os.getenv("MARKET_HOURS_END",   "16:00"), dt_time(16, 0))
    afterhours_end  = _parse_hhmm(os.getenv("AFTERHOURS_END",     "20:00"), dt_time(20, 0))

    if premarket_start <= current < market_open:
        return MarketPeriod.PREMARKET
    if market_open <= current < market_close:
        return MarketPeriod.REGULAR
    if market_close <= current < afterhours_end:
        return MarketPeriod.AFTER_HOURS
    return MarketPeriod.CLOSED


def is_etf_tradeable(now: datetime | None = None) -> bool:
    """
    Return True if ETF trading is currently permitted (any non-CLOSED period).

    Crypto spot positions trade 24/7 regardless of this flag.
    """
    return get_market_period(now) != MarketPeriod.CLOSED


def get_etf_order_type(now: datetime | None = None) -> str:
    """
    Return the Kraken order-type string appropriate for the current period.

    Regular hours → ``"mkt"``   (ORDER_TYPE_MARKET)
    Pre/after-hours → ``"limit"``  (ORDER_TYPE_LIMIT)
    Closed → ``"limit"`` (caller should skip the order entirely via
    ``is_etf_tradeable()`` but this provides a safe fallback).
    """
    period = get_market_period(now)
    if period == MarketPeriod.REGULAR:
        return ORDER_TYPE_MARKET
    return ORDER_TYPE_LIMIT


def describe_market_period(now: datetime | None = None) -> str:
    """Human-readable summary of the current market period and order type."""
    period     = get_market_period(now)
    order_type = get_etf_order_type(now)
    tradeable  = period != MarketPeriod.CLOSED
    return (
        f"period={period.value}  "
        f"tradeable={tradeable}  "
        f"order_type={order_type}"
    )
