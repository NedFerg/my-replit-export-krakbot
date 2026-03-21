"""
project/utils/constants.py
---------------------------
Centralised constants and helpers for ETF trading and order-type enforcement.

Importing from here avoids circular dependencies between broker.py and
utils/market_hours.py, and keeps order-type constants in one place.
"""

from __future__ import annotations

import enum

from utils.market_hours import etf_trading_allowed as _etf_trading_allowed
from utils.market_hours import required_order_type as _required_order_type

# ---------------------------------------------------------------------------
# Order-type string constants
# ---------------------------------------------------------------------------

ORDER_TYPE_MARKET: str = "market"
ORDER_TYPE_LIMIT: str = "limit"


# ---------------------------------------------------------------------------
# MarketPeriod — mirrors the MarketSession names for external consumers
# ---------------------------------------------------------------------------

class MarketPeriod(str, enum.Enum):
    """Named states of the US equity / ETF trading session."""
    CLOSED      = "closed"
    PREMARKET   = "premarket"
    REGULAR     = "regular"
    AFTER_HOURS = "after_hours"


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

def is_etf_tradeable(now=None) -> bool:
    """
    Return True if ETF orders are permitted at the given moment.

    Delegates to utils.market_hours.etf_trading_allowed().

    Parameters
    ----------
    now : optional datetime to evaluate; defaults to current Eastern time.
    """
    return _etf_trading_allowed(now)


def get_etf_order_type(now=None) -> str:
    """
    Return the appropriate Kraken order-type string for the current session.

    Regular hours → ORDER_TYPE_MARKET ("market")
    Extended hours → ORDER_TYPE_LIMIT  ("limit")

    Delegates to utils.market_hours.required_order_type().

    Parameters
    ----------
    now : optional datetime to evaluate; defaults to current Eastern time.
    """
    return _required_order_type(now)
