"""
BTC Breakout Detector
=====================
Returns a confidence score (0.0–1.0) that BTC has broken out above its ATH
and is sustaining the move.

Criteria weighted into the score:
  - Price above ATH threshold ($100k) and held for 3+ daily closes  → 0.50
  - Volume above 20-day average                                       → 0.30
  - Short-term momentum positive (5-bar return > 0)                  → 0.20

Usage::

    from strategies.signals.btc_breakout_detector import BTCBreakoutDetector

    detector = BTCBreakoutDetector()
    # Feed one price+volume observation per call (e.g. once per bar/minute):
    confidence = detector.update(btc_price=102_000, volume=12_500)
    # Returns float in [0, 1].  >= 0.6 is considered a confirmed breakout.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_ATH_TARGET: float = 100_000.0   # $ price level that constitutes a new ATH
ATH_HOLD_BARS: int    = 3           # consecutive bars above ATH needed
VOLUME_AVG_WINDOW: int = 20         # bars used for average-volume baseline
MOMENTUM_WINDOW: int   = 5          # bars used for momentum calculation


class BTCBreakoutDetector:
    """
    Stateful signal engine that maintains rolling price / volume history and
    emits a continuous confidence score for the BTC ATH breakout condition.

    Parameters
    ----------
    ath_target : USD price level treated as the ATH breakout threshold.
    ath_hold_bars : How many consecutive bars price must stay above ath_target.
    volume_window : Look-back window for computing average volume.
    momentum_window : Look-back window for computing price momentum.
    """

    def __init__(
        self,
        ath_target: float = BTC_ATH_TARGET,
        ath_hold_bars: int = ATH_HOLD_BARS,
        volume_window: int = VOLUME_AVG_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
    ) -> None:
        self.ath_target     = ath_target
        self.ath_hold_bars  = ath_hold_bars
        self.volume_window  = volume_window
        self.momentum_window = momentum_window

        self._prices:  deque[float] = deque(maxlen=max(volume_window, momentum_window) + 5)
        self._volumes: deque[float] = deque(maxlen=volume_window + 5)

        # How many consecutive bars BTC has been above ath_target
        self._bars_above_ath: int = 0

        # Last emitted confidence (useful for callers that poll without feeding new data)
        self.last_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, btc_price: float, volume: float = 0.0) -> float:
        """
        Feed the latest BTC price (and optionally volume) and return the
        updated breakout confidence score in [0, 1].

        Parameters
        ----------
        btc_price : Latest BTC/USD price.
        volume    : Volume for this bar (same units throughout; pass 0 to
                    skip volume scoring and rely only on price signals).

        Returns
        -------
        float — confidence in [0, 1].
        """
        self._prices.append(float(btc_price))
        if volume > 0:
            self._volumes.append(float(volume))

        score = 0.0

        # --- Component 1: ATH breakout + hold (weight 0.50) -----------
        if btc_price >= self.ath_target:
            self._bars_above_ath += 1
        else:
            self._bars_above_ath = 0

        if self._bars_above_ath >= self.ath_hold_bars:
            # Full credit at exactly ath_hold_bars; bonus for longer holds
            ath_score = min(1.0, self._bars_above_ath / self.ath_hold_bars)
            score += 0.50 * ath_score
        elif self._bars_above_ath > 0:
            # Partial credit for 1-2 bars above ATH
            score += 0.50 * (self._bars_above_ath / self.ath_hold_bars)

        # --- Component 2: Volume confirmation (weight 0.30) -----------
        if len(self._volumes) >= 2:
            avg_vol = sum(list(self._volumes)[:-1]) / (len(self._volumes) - 1)
            if avg_vol > 0 and self._volumes[-1] > avg_vol:
                vol_ratio = min(self._volumes[-1] / avg_vol, 3.0)  # cap at 3×
                score += 0.30 * min((vol_ratio - 1.0) / 2.0 + 0.5, 1.0)
        elif volume > 0:
            # Not enough history — give half credit
            score += 0.15

        # --- Component 3: Momentum (weight 0.20) ----------------------
        momentum = self._momentum()
        if momentum > 0:
            score += 0.20 * min(momentum / 0.05, 1.0)  # full credit at 5 % move

        self.last_confidence = round(min(score, 1.0), 4)
        return self.last_confidence

    def reset(self) -> None:
        """Clear all history and reset the detector."""
        self._prices.clear()
        self._volumes.clear()
        self._bars_above_ath = 0
        self.last_confidence = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _momentum(self) -> float:
        """Return n-bar price return (positive = up-trend)."""
        prices = list(self._prices)
        n = self.momentum_window
        if len(prices) < n + 1:
            return 0.0
        base = prices[-(n + 1)]
        if base <= 0:
            return 0.0
        return (prices[-1] - base) / base
