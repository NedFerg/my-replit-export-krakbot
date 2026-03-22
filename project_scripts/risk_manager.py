#!/usr/bin/env python3
"""
Risk Manager - Enforces position limits, notional caps, and daily drawdown protection.

Environment variables (all optional, have defaults):
  RISK_MAX_POSITION_SIZE_PCT  – max fraction of account per position  (default: 0.15)
  RISK_MAX_NOTIONAL_USD       – max USD value per single position      (default: 100)
  RISK_MAX_DAILY_DRAWDOWN_PCT – max fraction of account lost per day   (default: 0.10)
"""

import logging
import os
from datetime import datetime, date

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralised risk gate that every order must pass before execution.

    All limits are loaded from environment variables so they scale with
    the operator's account size without any code changes.
    """

    def __init__(self, total_capital: float):
        self.total_capital = total_capital

        # ── limits from env ──────────────────────────────────────────────────
        self.max_position_size_pct = float(
            os.environ.get("RISK_MAX_POSITION_SIZE_PCT", "0.15")
        )
        self.max_notional_usd = float(
            os.environ.get("RISK_MAX_NOTIONAL_USD", "100")
        )
        self.max_daily_drawdown_pct = float(
            os.environ.get("RISK_MAX_DAILY_DRAWDOWN_PCT", "0.10")
        )

        # ── daily drawdown tracking ──────────────────────────────────────────
        self._today: date = date.today()
        self._day_start_capital: float = total_capital
        self._daily_realized_loss: float = 0.0

        logger.info(
            "[RiskManager] Initialised | capital=%.2f | max_pos_pct=%.1f%% | "
            "max_notional=$%.2f | max_drawdown=%.1f%%",
            total_capital,
            self.max_position_size_pct * 100,
            self.max_notional_usd,
            self.max_daily_drawdown_pct * 100,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def update_capital(self, new_capital: float) -> None:
        """Call after each trade to keep the capital reference current."""
        self.total_capital = new_capital

    def record_loss(self, loss_usd: float) -> None:
        """Register a realised loss (positive number = money lost)."""
        self._roll_day_if_needed()
        if loss_usd > 0:
            self._daily_realized_loss += loss_usd
            logger.debug(
                "[RiskManager] Recorded loss $%.2f | daily total $%.2f",
                loss_usd,
                self._daily_realized_loss,
            )

    def check_buy(self, notional_usd: float, asset: str = "") -> tuple[bool, str]:
        """
        Run all risk checks for a proposed BUY order.

        Returns
        -------
        (allowed, reason)
            allowed – True if the trade may proceed.
            reason  – Human-readable explanation (empty string when allowed).
        """
        self._roll_day_if_needed()

        # 1. Max notional per position
        if notional_usd > self.max_notional_usd:
            msg = (
                f"[RiskManager] BLOCKED {asset} BUY: notional ${notional_usd:.2f} "
                f"> max ${self.max_notional_usd:.2f}"
            )
            logger.warning(msg)
            return False, msg

        # 2. Max position size as % of account
        max_allowed = self.total_capital * self.max_position_size_pct
        if notional_usd > max_allowed:
            msg = (
                f"[RiskManager] BLOCKED {asset} BUY: notional ${notional_usd:.2f} "
                f"> {self.max_position_size_pct*100:.0f}% of capital "
                f"(${max_allowed:.2f})"
            )
            logger.warning(msg)
            return False, msg

        # 3. Daily drawdown cap
        drawdown_allowed = self._day_start_capital * self.max_daily_drawdown_pct
        if self._daily_realized_loss >= drawdown_allowed:
            msg = (
                f"[RiskManager] BLOCKED {asset} BUY: daily loss "
                f"${self._daily_realized_loss:.2f} >= drawdown cap "
                f"${drawdown_allowed:.2f} ({self.max_daily_drawdown_pct*100:.0f}%)"
            )
            logger.warning(msg)
            return False, msg

        logger.debug(
            "[RiskManager] APPROVED %s BUY $%.2f (daily_loss=$%.2f)",
            asset,
            notional_usd,
            self._daily_realized_loss,
        )
        return True, ""

    def clamp_notional(self, requested_usd: float, asset: str = "") -> float:
        """
        Return the maximum allowable notional for this order, clamped to both
        the per-position cap and the account-size cap.  Never raises; always
        returns a non-negative value.
        """
        cap_pct = self.total_capital * self.max_position_size_pct
        clamped = min(requested_usd, self.max_notional_usd, cap_pct)
        if clamped < requested_usd:
            logger.info(
                "[RiskManager] %s order clamped $%.2f → $%.2f",
                asset,
                requested_usd,
                clamped,
            )
        return max(clamped, 0.0)

    def daily_drawdown_remaining(self) -> float:
        """Return how much USD loss budget remains for today."""
        self._roll_day_if_needed()
        cap = self._day_start_capital * self.max_daily_drawdown_pct
        return max(cap - self._daily_realized_loss, 0.0)

    def status(self) -> dict:
        """Return a snapshot of current risk state for logging."""
        self._roll_day_if_needed()
        return {
            "total_capital": self.total_capital,
            "max_position_pct": self.max_position_size_pct,
            "max_notional_usd": self.max_notional_usd,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "daily_loss_so_far": self._daily_realized_loss,
            "daily_loss_budget_remaining": self.daily_drawdown_remaining(),
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _roll_day_if_needed(self) -> None:
        """Reset daily counters when the calendar date changes."""
        today = date.today()
        if today != self._today:
            logger.info(
                "[RiskManager] New trading day – resetting daily loss counter "
                "(was $%.2f). Capital reference updated to $%.2f",
                self._daily_realized_loss,
                self.total_capital,
            )
            self._today = today
            self._day_start_capital = self.total_capital
            self._daily_realized_loss = 0.0
