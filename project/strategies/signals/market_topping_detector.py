"""
Market Topping Detector
=======================
Returns True when the overall crypto market shows distribution / topping signals.

Criteria (all must fire for a confirmed top signal):
  1. BTC lower high: current BTC price below its recent peak AND momentum broken
  2. Broad overbought: RSI > 80 on 5+ of the tracked alts
  3. Breadth deterioration: more assets declining than advancing over the window

Usage::

    from strategies.signals.market_topping_detector import MarketToppingDetector

    det = MarketToppingDetector(assets=["SOL", "XRP", "HBAR", "LINK", "XLM"])
    is_top = det.update(prices={"BTC": 105_000, "SOL": 200, ...})
"""

from __future__ import annotations

from collections import deque

from strategies.signals.indicators import rsi as _rsi_fn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RSI_OVERBOUGHT: int   = 80
MIN_OVERBOUGHT_ASSETS: int = 5       # alts with RSI > 80 needed to confirm
BREADTH_WINDOW: int   = 3            # bars to compare for advance/decline breadth
MOMENTUM_WINDOW: int  = 5
BTC_PEAK_LOOKBACK: int = 10          # bars to look for BTC recent high


class MarketToppingDetector:
    """
    Monitors multi-asset RSI, BTC price action, and market breadth to flag
    when the market is likely topping and a bear phase may be beginning.

    Parameters
    ----------
    assets       : List of alt tickers to monitor (not including BTC).
    rsi_overbought   : RSI threshold above which an asset is considered overbought.
    min_overbought   : How many assets must be overbought to fire criterion 2.
    breadth_window   : Bars over which to measure advance vs. decline count.
    btc_peak_lookback: Bars to look back for BTC recent high.
    """

    def __init__(
        self,
        assets: list[str] | None = None,
        rsi_overbought: int   = RSI_OVERBOUGHT,
        min_overbought: int   = MIN_OVERBOUGHT_ASSETS,
        breadth_window: int   = BREADTH_WINDOW,
        btc_peak_lookback: int = BTC_PEAK_LOOKBACK,
        momentum_window: int  = MOMENTUM_WINDOW,
    ) -> None:
        self.assets          = assets or ["SOL", "XRP", "HBAR", "LINK", "XLM"]
        self.rsi_overbought  = rsi_overbought
        self.min_overbought  = min_overbought
        self.breadth_window  = breadth_window
        self.btc_peak_lookback = btc_peak_lookback
        self.momentum_window = momentum_window

        maxlen = max(btc_peak_lookback, 30) + 5
        self._btc_prices:    deque[float]              = deque(maxlen=maxlen)
        self._asset_prices:  dict[str, deque[float]]   = {
            a: deque(maxlen=maxlen) for a in self.assets
        }

        # Cached result
        self.last_result: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prices: dict[str, float]) -> bool:
        """
        Feed the latest price snapshot for all tracked assets and return
        whether market-topping conditions are met.

        Parameters
        ----------
        prices : Dict of {ticker: price}.  Must include "BTC" plus the alts.

        Returns
        -------
        bool — True if market topping signal fires.
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

        # --- Criterion 1: BTC lower high + momentum broken -------------
        if not self._btc_lower_high(btc):
            return False

        # --- Criterion 2: 5+ alts with RSI > 80 -----------------------
        overbought_count = 0
        for asset in self.assets:
            prices = list(self._asset_prices[asset])
            rsi_val = _rsi_fn(prices)
            if rsi_val is not None and rsi_val > self.rsi_overbought:
                overbought_count += 1

        if overbought_count < self.min_overbought:
            return False

        # --- Criterion 3: breadth deterioration -----------------------
        if not self._breadth_deteriorating():
            return False

        return True

    def _btc_lower_high(self, btc: list[float]) -> bool:
        """BTC is making a lower high and momentum is broken."""
        n = self.btc_peak_lookback
        if len(btc) < n + 1:
            return False

        recent_high = max(btc[-n:])
        prior_high  = max(btc[-(n * 2):-n]) if len(btc) >= n * 2 else recent_high

        # Lower high: current peak is below prior peak
        if recent_high >= prior_high:
            return False

        # Momentum broken: latest price below short-term average
        window = self.momentum_window
        if len(btc) >= window:
            avg = sum(btc[-window:]) / window
            if btc[-1] >= avg:
                return False  # price still above average — no breakdown

        return True

    def _breadth_deteriorating(self) -> bool:
        """More assets declining than advancing over breadth_window bars."""
        w = self.breadth_window
        advancing = 0
        declining = 0
        for asset in self.assets:
            prices = list(self._asset_prices[asset])
            if len(prices) < w + 1:
                continue
            change = (prices[-1] - prices[-(w + 1)]) / max(prices[-(w + 1)], 1e-9)
            if change > 0:
                advancing += 1
            elif change < 0:
                declining += 1

        return declining > advancing
