#!/usr/bin/env python3
"""
Portfolio Manager - Tracks capital, PnL, and drives exponential growth
through automatic profit reinvestment.

Works at ANY capital level: $1, $10, $50, $100, $1k, $10k, $50k+
No hardcoded minimums.
"""

import logging
import os
from datetime import datetime, date

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration (all values have sensible defaults)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to default."""
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Bull Market Mode — set BULL_MARKET_MODE=true in .env to enable
# ---------------------------------------------------------------------------
# When enabled, two risk defaults are relaxed to maximise compounding during a
# sustained bull market.  Individual env-var overrides still take precedence.
#
#   RISK_MAX_DAILY_DRAWDOWN_PCT   0.10 → 0.18  (tolerate larger intra-day swings)
#   REINVESTMENT_PROFIT_THRESHOLD_PCT  0.25 → 0.10  (faster compounding)
#
# Existing safety checks (kill-switch, position caps, API guards) are unchanged.
BULL_MARKET_MODE: bool = os.getenv("BULL_MARKET_MODE", "false").lower() in ("1", "true", "yes")

# Total capital the bot is allowed to deploy (can be ANY positive amount).
# The bot will NEVER refuse to trade because this value is "too small".
TOTAL_TRADING_CAPITAL: float = _env_float("TOTAL_TRADING_CAPITAL", 50.0)

# Fraction of capital to risk on a single trade (e.g. 0.15 = 15%).
RISK_MAX_POSITION_SIZE_PCT: float = _env_float("RISK_MAX_POSITION_SIZE_PCT", 0.15)

# Max USD notional per trade.  0 = no hard cap (derive from instance capital).
_RAW_MAX_NOTIONAL: float = _env_float("RISK_MAX_NOTIONAL_USD", 0.0)
# Do NOT derive a module-level default here — the instance capital is unknown
# at import time.  PortfolioManager.__init__ sets the cap from its own capital.
RISK_MAX_NOTIONAL_USD: float = _RAW_MAX_NOTIONAL  # 0 means "no cap"

# Max daily drawdown before trading is paused for the day (e.g. 0.10 = 10%).
# Bull mode raises this to 0.18 (18%) to tolerate larger intra-day swings.
_DEFAULT_DAILY_DD: float = 0.18 if BULL_MARKET_MODE else 0.10
RISK_MAX_DAILY_DRAWDOWN_PCT: float = _env_float("RISK_MAX_DAILY_DRAWDOWN_PCT", _DEFAULT_DAILY_DD)

# Reinvest when realised profits reach this fraction of starting capital
# (e.g. 0.25 = reinvest after every 25% profit milestone).
# Bull mode lowers this to 0.10 (10%) for faster compounding.
_DEFAULT_REINVEST: float = 0.10 if BULL_MARKET_MODE else 0.25
REINVESTMENT_PROFIT_THRESHOLD_PCT: float = _env_float(
    "REINVESTMENT_PROFIT_THRESHOLD_PCT", _DEFAULT_REINVEST
)


class PortfolioManager:
    """
    Tracks overall portfolio health and drives exponential growth through
    profit reinvestment.

    Design principles
    -----------------
    * Works at any starting capital — $1 behaves the same as $50 000.
    * capital_per_trade grows automatically as realised profits accumulate.
    * Daily drawdown protection pauses new trades without stopping the bot.
    * All state is in-memory; no external DB required.
    """

    def __init__(
        self,
        total_capital: float | None = None,
        risk_pct: float | None = None,
        max_notional: float | None = None,
        max_daily_drawdown_pct: float | None = None,
        reinvestment_threshold_pct: float | None = None,
    ):
        self.starting_capital: float = total_capital if total_capital is not None else TOTAL_TRADING_CAPITAL
        self.total_capital: float = self.starting_capital  # grows with profits

        self.risk_pct: float = risk_pct if risk_pct is not None else RISK_MAX_POSITION_SIZE_PCT
        # max_notional: use explicit arg, or env override, or derive from instance capital
        if max_notional is not None:
            self.max_notional: float = max_notional
        elif RISK_MAX_NOTIONAL_USD > 0:
            self.max_notional = RISK_MAX_NOTIONAL_USD
        else:
            # 0 = no hard cap; capital_per_trade() uses risk_pct directly
            self.max_notional = 0.0
        self.max_daily_drawdown_pct: float = (
            max_daily_drawdown_pct
            if max_daily_drawdown_pct is not None
            else RISK_MAX_DAILY_DRAWDOWN_PCT
        )
        self.reinvestment_threshold_pct: float = (
            reinvestment_threshold_pct
            if reinvestment_threshold_pct is not None
            else REINVESTMENT_PROFIT_THRESHOLD_PCT
        )

        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0

        # Daily drawdown tracking
        self._day_start_capital: float = self.starting_capital
        self._current_day: date = datetime.utcnow().date()
        self._daily_trading_paused: bool = False

        # Reinvestment tracking
        self._last_reinvestment_capital: float = self.starting_capital
        self._reinvestment_count: int = 0

        logger.info(
            "[PortfolioManager] Initialised | capital=$%.2f | risk=%.0f%% | "
            "max_notional=$%.2f | daily_dd_limit=%.0f%%",
            self.total_capital,
            self.risk_pct * 100,
            self.max_notional,
            self.max_daily_drawdown_pct * 100,
        )
        if BULL_MARKET_MODE:
            logger.warning(
                "[PortfolioManager] ⚠️  Bull Market Mode enabled: risk and ETF allocation "
                "limits relaxed; see README.md for details. "
                "daily_dd_limit=%.0f%% reinvestment_threshold=%.0f%%",
                self.max_daily_drawdown_pct * 100,
                self.reinvestment_threshold_pct * 100,
            )

    # ------------------------------------------------------------------
    # Capital sizing
    # ------------------------------------------------------------------

    def capital_per_trade(self) -> float:
        """
        Return the USD amount to allocate to the next trade.

        Scales proportionally with total_capital so the same strategy
        works at $1, $100, $10 000, and $50 000 without any code change.
        The result is always >= 0 (it is the caller's responsibility to
        decide whether such a tiny amount can be sent to the exchange).
        """
        raw = self.total_capital * self.risk_pct
        # Cap at max notional if configured
        capped = min(raw, self.max_notional) if self.max_notional > 0 else raw
        return max(capped, 0.0)

    # ------------------------------------------------------------------
    # Daily drawdown protection
    # ------------------------------------------------------------------

    def _reset_day_if_needed(self) -> None:
        today = datetime.utcnow().date()
        if today != self._current_day:
            self._current_day = today
            self._day_start_capital = self.total_capital
            self._daily_trading_paused = False
            logger.info(
                "[PortfolioManager] New trading day — daily drawdown reset. "
                "Day-start capital=$%.2f",
                self._day_start_capital,
            )

    def daily_drawdown_breached(self) -> bool:
        """Return True if the daily loss limit has been reached."""
        self._reset_day_if_needed()
        if self._day_start_capital <= 0:
            return False
        drawdown = (self._day_start_capital - self.total_capital) / self._day_start_capital
        if drawdown >= self.max_daily_drawdown_pct:
            if not self._daily_trading_paused:
                self._daily_trading_paused = True
                logger.warning(
                    "[PortfolioManager] Daily drawdown limit reached "
                    "(%.1f%% >= %.1f%%). Trading paused until tomorrow.",
                    drawdown * 100,
                    self.max_daily_drawdown_pct * 100,
                )
            return True
        return False

    # ------------------------------------------------------------------
    # Trade lifecycle hooks
    # ------------------------------------------------------------------

    def on_trade_open(self, usd_allocated: float) -> None:
        """Call when a new trade is opened (reserves capital)."""
        self._reset_day_if_needed()
        logger.debug("[PortfolioManager] Trade opened | allocated=$%.2f", usd_allocated)

    def on_trade_close(self, realized_profit: float) -> None:
        """
        Call when a trade is closed.

        Parameters
        ----------
        realized_profit : float
            Signed PnL of the closed trade in USD.  Positive = profit,
            negative = loss.
        """
        self._reset_day_if_needed()
        self.realized_pnl += realized_profit
        self.total_capital += realized_profit

        logger.info(
            "[PortfolioManager] Trade closed | PnL=$%+.2f | "
            "total_realized=$%+.2f | total_capital=$%.2f",
            realized_profit,
            self.realized_pnl,
            self.total_capital,
        )

        self._check_reinvestment()

    def update_unrealized(self, unrealized_pnl: float) -> None:
        """Update the aggregate unrealized PnL (call each iteration)."""
        self.unrealized_pnl = unrealized_pnl

    # ------------------------------------------------------------------
    # Exponential growth: profit reinvestment
    # ------------------------------------------------------------------

    def _check_reinvestment(self) -> None:
        """
        Trigger reinvestment when realised profits cross the threshold.

        After each reinvestment, the baseline resets so the next milestone
        is measured from the newly grown capital — this creates compounding.
        """
        if self._last_reinvestment_capital <= 0:
            return

        profit_since_last = self.total_capital - self._last_reinvestment_capital
        threshold = self._last_reinvestment_capital * self.reinvestment_threshold_pct

        if profit_since_last >= threshold:
            self._reinvestment_count += 1
            old_cpt = self._last_reinvestment_capital * self.risk_pct
            new_cpt = self.total_capital * self.risk_pct
            self._last_reinvestment_capital = self.total_capital

            # max_notional scales with capital when an explicit cap was set
            if self.max_notional > 0:
                self.max_notional = self.total_capital * self.risk_pct

            logger.info(
                "[PortfolioManager] 🚀 REINVESTMENT #%d triggered | "
                "capital=$%.2f (+%.1f%%) | capital_per_trade: $%.2f → $%.2f",
                self._reinvestment_count,
                self.total_capital,
                (self.total_capital / self.starting_capital - 1) * 100,
                old_cpt,
                new_cpt,
            )

    # ------------------------------------------------------------------
    # Portfolio risk check (call before placing any order)
    # ------------------------------------------------------------------

    # Fraction of tolerance above capital_per_trade() that is still approved.
    # Accommodates minor floating-point rounding without blocking valid trades.
    _POSITION_SIZE_TOLERANCE = 0.05  # 5%

    def check_trade_allowed(self, proposed_usd: float) -> tuple[bool, str]:
        """
        Return (allowed, reason) for a proposed trade.

        Parameters
        ----------
        proposed_usd : float
            USD notional of the proposed trade.

        Returns
        -------
        (True, "ok")              – trade is allowed
        (False, "<reason>")       – trade is blocked; reason explains why
        """
        self._reset_day_if_needed()

        if self._daily_trading_paused or self.daily_drawdown_breached():
            return False, "daily_drawdown_limit"

        max_allowed = self.capital_per_trade()
        if proposed_usd > max_allowed * (1 + self._POSITION_SIZE_TOLERANCE):
            return False, (
                f"position_too_large: ${proposed_usd:.2f} > ${max_allowed:.2f}"
            )

        if self.total_capital <= 0:
            return False, "zero_capital"

        return True, "ok"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def growth_pct(self) -> float:
        """Total portfolio growth since inception (%)."""
        if self.starting_capital <= 0:
            return 0.0
        return (self.total_capital / self.starting_capital - 1) * 100

    def summary(self) -> dict:
        """Return a dict suitable for logging."""
        return {
            "starting_capital": round(self.starting_capital, 2),
            "total_capital": round(self.total_capital, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "growth_pct": round(self.growth_pct(), 2),
            "capital_per_trade": round(self.capital_per_trade(), 2),
            "reinvestment_count": self._reinvestment_count,
            "daily_trading_paused": self._daily_trading_paused,
        }

    def log_summary(self) -> None:
        """Print a formatted summary to the logger."""
        s = self.summary()
        logger.info(
            "[PortfolioManager] capital=$%.2f | growth=%.1f%% | "
            "realized=$%+.2f | unrealized=$%+.2f | "
            "capital_per_trade=$%.2f | reinvestments=%d",
            s["total_capital"],
            s["growth_pct"],
            s["realized_pnl"],
            s["unrealized_pnl"],
            s["capital_per_trade"],
            s["reinvestment_count"],
        )
