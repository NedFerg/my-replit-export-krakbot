"""
project/utils/market_hours.py
------------------------------
Market hours detection and order-type enforcement for ETF hedging.

US equity / ETF markets trade on the following schedule (all times Eastern):
  Pre-market   : 04:00 – 09:29  → limit orders only
  Regular hours: 09:30 – 15:59  → market orders permitted
  After-hours  : 16:00 – 19:59  → limit orders only
  Closed       : 20:00 – 03:59  → no ETF orders

Crypto spot markets trade 24/7 and are completely unaffected by this module.

Usage
-----
    from utils.market_hours import MarketHours

    mh = MarketHours()
    if mh.etf_trading_allowed():
        order_type = mh.required_order_type()   # "market" or "limit"
"""

import datetime
import os
import zoneinfo

# ---------------------------------------------------------------------------
# Configurable via environment variables
# ---------------------------------------------------------------------------

_TZ_NAME = os.getenv("TRADING_TIMEZONE", "America/New_York")
_ALLOW_PREMARKET    = os.getenv("ALLOW_PREMARKET",    "true").lower() not in ("false", "0", "no")
_ALLOW_AFTER_HOURS  = os.getenv("ALLOW_AFTER_HOURS",  "true").lower() not in ("false", "0", "no")

# After-hours session ends at 20:00 ET by default (configurable)
_AFTER_HOURS_END_HOUR = int(os.getenv("AFTER_HOURS_END_HOUR", "20"))

try:
    _EASTERN = zoneinfo.ZoneInfo(_TZ_NAME)
except zoneinfo.ZoneInfoNotFoundError:
    # Fallback: use UTC offset for US Eastern time (no DST awareness)
    _EASTERN = datetime.timezone(datetime.timedelta(hours=-5))


class MarketSession:
    """Named constants for the current market session."""
    CLOSED      = "closed"
    PREMARKET   = "premarket"
    REGULAR     = "regular"
    AFTER_HOURS = "after_hours"


class MarketHours:
    """
    Stateless helper that queries the current wall-clock time (Eastern) and
    returns the active session and the required order type for ETF trades.

    All public methods accept an optional `now` parameter (datetime) so they
    can be unit-tested without mocking time.  In production, `now` defaults
    to the current time in the configured timezone.
    """

    # Session boundaries (hour, minute) in Eastern time
    PREMARKET_START    = (4,  0)
    REGULAR_START      = (9, 30)
    AFTER_HOURS_START  = (16, 0)
    AFTER_HOURS_END    = (_AFTER_HOURS_END_HOUR, 0)

    def _now_eastern(self) -> datetime.datetime:
        """Return the current time localised to the Eastern timezone."""
        return datetime.datetime.now(tz=_EASTERN)

    def _to_eastern(self, dt: datetime.datetime) -> datetime.datetime:
        """Attach Eastern timezone to a naïve datetime, or convert an aware one."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_EASTERN)
        return dt.astimezone(_EASTERN)

    def get_session(self, now: datetime.datetime | None = None) -> str:
        """
        Return the current MarketSession constant for the given moment.

        Parameters
        ----------
        now : datetime to evaluate.  Defaults to current Eastern time.

        Returns
        -------
        One of MarketSession.{CLOSED, PREMARKET, REGULAR, AFTER_HOURS}.
        """
        et = self._to_eastern(now) if now is not None else self._now_eastern()

        # Weekends: markets closed
        if et.weekday() >= 5:   # 5 = Saturday, 6 = Sunday
            return MarketSession.CLOSED

        hm = (et.hour, et.minute)

        if hm < self.PREMARKET_START:
            return MarketSession.CLOSED
        if hm < self.REGULAR_START:
            return MarketSession.PREMARKET
        if hm < self.AFTER_HOURS_START:
            return MarketSession.REGULAR
        if hm < self.AFTER_HOURS_END:
            return MarketSession.AFTER_HOURS
        return MarketSession.CLOSED

    def etf_trading_allowed(self, now: datetime.datetime | None = None) -> bool:
        """
        Return True if ETF orders are permitted at the given moment.

        Pre-market and after-hours sessions are enabled/disabled via the
        ALLOW_PREMARKET and ALLOW_AFTER_HOURS environment variables.
        Regular hours are always permitted.
        Closed (overnight / weekends) is always blocked.

        Parameters
        ----------
        now : datetime to evaluate.  Defaults to current Eastern time.
        """
        session = self.get_session(now)
        if session == MarketSession.REGULAR:
            return True
        if session == MarketSession.PREMARKET:
            return _ALLOW_PREMARKET
        if session == MarketSession.AFTER_HOURS:
            return _ALLOW_AFTER_HOURS
        # CLOSED
        return False

    def required_order_type(self, now: datetime.datetime | None = None) -> str:
        """
        Return the Kraken order-type string appropriate for the current session.

        Regular hours → "market"  (fastest execution; allowed by exchange)
        Extended hours → "limit"   (market orders rejected outside regular hours)

        Parameters
        ----------
        now : datetime to evaluate.  Defaults to current Eastern time.

        Returns
        -------
        "market" or "limit"
        """
        session = self.get_session(now)
        if session == MarketSession.REGULAR:
            return "market"
        return "limit"

    def status_line(self, now: datetime.datetime | None = None) -> str:
        """Return a compact human-readable status string for logging."""
        et      = self._to_eastern(now) if now is not None else self._now_eastern()
        session = self.get_session(now)
        allowed = self.etf_trading_allowed(now)
        otype   = self.required_order_type(now) if allowed else "n/a"
        return (
            f"[MarketHours] {et.strftime('%Y-%m-%d %H:%M:%S %Z')}  "
            f"session={session}  etf_allowed={allowed}  order_type={otype}"
        )


# ---------------------------------------------------------------------------
# Module-level singleton (convenience)
# ---------------------------------------------------------------------------

_market_hours = MarketHours()


def get_session(now: datetime.datetime | None = None) -> str:
    """Module-level convenience wrapper for MarketHours.get_session()."""
    return _market_hours.get_session(now)


def etf_trading_allowed(now: datetime.datetime | None = None) -> bool:
    """Module-level convenience wrapper for MarketHours.etf_trading_allowed()."""
    return _market_hours.etf_trading_allowed(now)


def required_order_type(now: datetime.datetime | None = None) -> str:
    """Module-level convenience wrapper for MarketHours.required_order_type()."""
    return _market_hours.required_order_type(now)


# Backward compatibility aliases for older broker code
def is_etf_tradeable(now=None):
    """Alias for etf_trading_allowed()."""
    return etf_trading_allowed(now)


def get_etf_order_type(now=None):
    """Alias for required_order_type()."""
    return required_order_type(now)


# Older constant names
MarketPeriod = MarketSession
ORDER_TYPE_MARKET = "market"
ORDER_TYPE_LIMIT = "limit"
