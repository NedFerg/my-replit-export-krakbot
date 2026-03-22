#!/usr/bin/env python3
"""
Portfolio Manager - Tracks capital, realized/unrealized PnL, and drives
exponential growth by auto-reinvesting profits into larger position sizes.

Environment variables (all optional, have defaults):
  TOTAL_TRADING_CAPITAL            – starting capital in USD          (default: 50)
  REINVESTMENT_PROFIT_THRESHOLD_PCT – profit % that triggers reinvest (default: 0.25)
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Tracks the total portfolio and scales capital_per_trade automatically
    as realized profits accumulate.

    Exponential growth loop
    -----------------------
    1. Bot trades with `capital_per_trade` USD per position.
    2. After each SELL the realized profit is added to `realized_profits`.
    3. When `realized_profits / base_capital >= reinvestment_threshold`,
       `capital_per_trade` is increased to include a share of accumulated
       profits → the next trade is bigger → profits compound.
    """

    def __init__(self, base_capital: float | None = None):
        # Base capital from env or argument
        self.base_capital: float = base_capital or float(
            os.environ.get("TOTAL_TRADING_CAPITAL", "50")
        )

        # Reinvestment trigger: default 25%
        self.reinvestment_threshold: float = float(
            os.environ.get("REINVESTMENT_PROFIT_THRESHOLD_PCT", "0.25")
        )

        # Running totals
        self.realized_profits: float = 0.0
        self.unrealized_pnl: float = 0.0

        # Start capital_per_trade at a safe fraction of base capital.
        # The runner may also pass a value explicitly via update_capital_per_trade().
        self.capital_per_trade: float = self.base_capital

        # How many times we have reinvested (for logging / auditing)
        self.reinvestment_count: int = 0

        # History for auditing
        self._events: list[dict] = []

        logger.info(
            "[PortfolioManager] Initialised | base_capital=$%.2f | "
            "reinvest_threshold=%.0f%% | capital_per_trade=$%.2f",
            self.base_capital,
            self.reinvestment_threshold * 100,
            self.capital_per_trade,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def record_trade_profit(self, realized_pnl: float, asset: str = "") -> None:
        """
        Record a closed trade's realized PnL (can be negative for a loss).
        Triggers a reinvestment check after every profitable trade.
        """
        self.realized_profits += realized_pnl
        self._log_event("TRADE_CLOSED", asset=asset, pnl=realized_pnl)

        logger.info(
            "[PortfolioManager] %s trade closed | PnL $%+.2f | "
            "total realized profits $%+.2f | portfolio growth %.1f%%",
            asset,
            realized_pnl,
            self.realized_profits,
            self.portfolio_growth_pct(),
        )

        if realized_pnl > 0:
            self._maybe_reinvest()

    def update_unrealized(self, total_unrealized_usd: float) -> None:
        """Update the aggregate unrealized PnL across all open positions."""
        self.unrealized_pnl = total_unrealized_usd

    def update_capital_per_trade(self, new_value: float) -> None:
        """Manually override capital_per_trade (e.g. from the runner on startup)."""
        self.capital_per_trade = new_value
        logger.info(
            "[PortfolioManager] capital_per_trade set to $%.2f", new_value
        )

    def total_equity(self) -> float:
        """Estimated total equity = base + realized profits + unrealized."""
        return self.base_capital + self.realized_profits + self.unrealized_pnl

    def portfolio_growth_pct(self) -> float:
        """Portfolio growth as a percentage of the original base capital."""
        if self.base_capital == 0:
            return 0.0
        return (self.realized_profits / self.base_capital) * 100

    def snapshot(self) -> dict:
        """Return a full portfolio snapshot for logging."""
        return {
            "base_capital": self.base_capital,
            "realized_profits": self.realized_profits,
            "unrealized_pnl": self.unrealized_pnl,
            "total_equity": self.total_equity(),
            "portfolio_growth_pct": self.portfolio_growth_pct(),
            "capital_per_trade": self.capital_per_trade,
            "reinvestment_count": self.reinvestment_count,
            "reinvestment_threshold_pct": self.reinvestment_threshold * 100,
        }

    def log_snapshot(self) -> None:
        """Emit a structured INFO log of the current portfolio state."""
        s = self.snapshot()
        logger.info("")
        logger.info("[PortfolioManager] ── Portfolio Snapshot ─────────────────")
        logger.info("  Base capital:        $%.2f", s["base_capital"])
        logger.info("  Realized profits:    $%+.2f", s["realized_profits"])
        logger.info("  Unrealized PnL:      $%+.2f", s["unrealized_pnl"])
        logger.info("  Total equity:        $%.2f", s["total_equity"])
        logger.info("  Portfolio growth:    %.1f%%", s["portfolio_growth_pct"])
        logger.info("  Capital per trade:   $%.2f", s["capital_per_trade"])
        logger.info(
            "  Reinvestments:       %d (threshold=%.0f%%)",
            s["reinvestment_count"],
            s["reinvestment_threshold_pct"],
        )
        logger.info("[PortfolioManager] ──────────────────────────────────────")
        logger.info("")

    # ── private helpers ───────────────────────────────────────────────────────

    def _maybe_reinvest(self) -> None:
        """
        If total realized profit has grown by at least `reinvestment_threshold`
        relative to base_capital since the last reinvestment, increase
        capital_per_trade proportionally.
        """
        growth_pct = self.portfolio_growth_pct()
        threshold_crossed = (
            growth_pct >= self.reinvestment_threshold * 100 * (self.reinvestment_count + 1)
        )

        if not threshold_crossed:
            return

        # Compound capital_per_trade: base + (profit * threshold fraction per step)
        old_cpt = self.capital_per_trade
        # Each reinvestment step adds one threshold-fraction of profits on top of
        # the previous capital_per_trade, producing true compounding.
        profit_increment = self.realized_profits * self.reinvestment_threshold
        self.capital_per_trade += profit_increment
        self.reinvestment_count += 1

        self._log_event(
            "REINVESTMENT",
            old_capital_per_trade=old_cpt,
            new_capital_per_trade=self.capital_per_trade,
            growth_pct=growth_pct,
        )

        logger.info(
            "[PortfolioManager] *** REINVESTMENT #%d triggered! "
            "Portfolio growth=%.1f%% | capital_per_trade $%.2f → $%.2f ***",
            self.reinvestment_count,
            growth_pct,
            old_cpt,
            self.capital_per_trade,
        )

    def _log_event(self, event_type: str, **kwargs) -> None:
        self._events.append(
            {"ts": datetime.utcnow().isoformat(), "event": event_type, **kwargs}
        )
