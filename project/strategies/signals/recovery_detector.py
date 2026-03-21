"""
Recovery Detector
=================
Returns True when the market shows early recovery signals after a bear phase,
indicating it may be safe to exit short positions and re-enter longs.

Criteria (both must fire):
  1. BTC higher low: BTC prints a trough higher than the previous trough,
     AND short-term momentum is rebuilding (5-bar return turning positive).
  2. Breadth improving: more assets advancing than declining over the window.

Usage::

    from strategies.signals.recovery_detector import RecoveryDetector

    det = RecoveryDetector()
    recovering = det.update(prices={"BTC": 90_000, "SOL": 140, ...})
"""

from __future__ import annotations

from collections import deque


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TROUGH_LOOKBACK: int  = 10     # bars used to identify the most recent trough
BREADTH_WINDOW: int   = 3      # bars over which to compute advance/decline
MOMENTUM_WINDOW: int  = 5      # bars for momentum calculation


class RecoveryDetector:
    """
    Monitors BTC price action and multi-asset breadth to flag when the market
    is showing early recovery signals after a bear drawdown.

    Parameters
    ----------
    assets          : Alt tickers to include in breadth measurement.
    trough_lookback : Bars used to identify recent vs. prior trough.
    breadth_window  : Bars for advance/decline breadth measurement.
    momentum_window : Bars for short-term momentum.
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        trough_lookback: int  = TROUGH_LOOKBACK,
        breadth_window: int   = BREADTH_WINDOW,
        momentum_window: int  = MOMENTUM_WINDOW,
    ) -> None:
        self.assets          = assets or ["SOL", "XRP", "HBAR", "LINK", "XLM"]
        self.trough_lookback = trough_lookback
        self.breadth_window  = breadth_window
        self.momentum_window = momentum_window

        maxlen = trough_lookback * 2 + 10
        self._btc_prices:   deque[float]            = deque(maxlen=maxlen)
        self._asset_prices: dict[str, deque[float]] = {
            a: deque(maxlen=maxlen) for a in self.assets
        }

        self.last_result: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prices: dict[str, float]) -> bool:
        """
        Feed the latest price snapshot and return whether recovery conditions
        are met.

        Parameters
        ----------
        prices : Dict of {ticker: price}.  Should include "BTC" and the alts.

        Returns
        -------
        bool — True if recovery signal fires.
        """
        btc_price = prices.get("BTC", 0.0)
        if btc_price > 0:
            self._btc_prices.append(float(btc_price))

        for asset in self.assets:
            p = prices.get(asset, 0.0)
            if p > 0:
                self._asset_prices[asset].append(float(p))

        self.last_result = self._evaluate()
        return self.last_result

    def reset(self) -> None:
        self._btc_prices.clear()
        for dq in self._asset_prices.values():
            dq.clear()
        self.last_result = False

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _evaluate(self) -> bool:
        btc = list(self._btc_prices)
        n = self.trough_lookback

        # Need enough data to compare two trough windows
        if len(btc) < n * 2:
            return False

        # --- Criterion 1: BTC higher low + momentum rebuilding ---------
        if not self._btc_higher_low(btc, n):
            return False

        # --- Criterion 2: breadth improving ----------------------------
        if not self._breadth_improving():
            return False

        return True

    def _btc_higher_low(self, btc: list[float], n: int) -> bool:
        """Check that the recent trough is higher than the prior trough and
        that short-term momentum is positive."""
        recent_trough = min(btc[-n:])
        prior_trough  = min(btc[-(n * 2):-n])

        if recent_trough <= prior_trough:
            return False  # still making lower lows

        # Momentum rebuilding: latest price above short-term average
        w = self.momentum_window
        if len(btc) >= w:
            avg = sum(btc[-w:]) / w
            if btc[-1] < avg:
                return False  # price still below short-term avg

        return True

    def _breadth_improving(self) -> bool:
        """More assets advancing than declining."""
        w = self.breadth_window
        advancing = 0
        declining  = 0
        for asset in self.assets:
            prices = list(self._asset_prices[asset])
            if len(prices) < w + 1:
                continue
            change = (prices[-1] - prices[-(w + 1)]) / max(prices[-(w + 1)], 1e-9)
            if change > 0:
                advancing += 1
            elif change < 0:
                declining += 1

        return advancing > declining
