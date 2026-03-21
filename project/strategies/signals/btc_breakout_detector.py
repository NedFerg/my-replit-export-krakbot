"""
BTC Breakout Detector
=====================
Returns a confidence score (0.0–1.0) that BTC is in a sustained bull run
and ready to drive an altcoin season.

Four complementary components drive the score (total weight = 1.00):

  Component 1 — Absolute ATH level (weight 0.20)
    Scores when price has been above ``ath_target`` for ``ath_hold_bars``
    consecutive bars.  Provides the strongest single signal when BTC is
    printing genuine new all-time highs (e.g. above $100 K).  Set
    ``BTC_ATH_TARGET`` env var to override the default.

  Component 2 — Rolling-high breakout (weight 0.30)   ← KEY for $74 K entry
    Scores when the current price is above the rolling maximum of the
    previous ``rolling_high_window`` bars, held for ``rolling_high_hold``
    consecutive bars.  This makes the detector **price-level agnostic**:
    if BTC is consistently making new local highs from $74 K upward it
    scores the same as it would at any other price level.  The bot
    therefore starts trading from wherever BTC happens to be when it
    connects to Kraken — it does NOT require a hard-coded $100 K threshold.

  Component 3 — Volume confirmation (weight 0.25)
    Current bar volume above the rolling average.

  Component 4 — Price momentum (weight 0.25)
    5-bar return positive; full credit at ≥ 5% move.

Typical scores from $74 K BTC in a strong uptrend:
  ATH level   : 0.00  (below $100 K)
  Rolling high: 0.30  (consistently printing new local highs)
  Volume      : ~0.20 (elevated but not extreme)
  Momentum    : ~0.15 (steady upward drift)
  Total       : ~0.65 → exceeds 0.55 threshold → triggers bull_alt_season

Usage::

    from strategies.signals.btc_breakout_detector import BTCBreakoutDetector

    detector = BTCBreakoutDetector()
    # Feed one price+volume observation per call (e.g. once per bar/second):
    confidence = detector.update(btc_price=74_000, volume=12_500)
    # Returns float in [0, 1].  >= 0.55 is considered a confirmed bull run.
"""

from __future__ import annotations

from collections import deque


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BTC_ATH_TARGET: float     = 100_000.0  # $ absolute ATH level (override via env var)
ATH_HOLD_BARS: int        = 3          # consecutive bars above ath_target needed
VOLUME_AVG_WINDOW: int    = 20         # bars for average-volume baseline
MOMENTUM_WINDOW: int      = 5          # bars for momentum calculation
ROLLING_HIGH_WINDOW: int  = 60         # bars for rolling-max comparison
ROLLING_HIGH_HOLD: int    = 2          # consecutive new-high bars for full score


class BTCBreakoutDetector:
    """
    Stateful signal engine that emits a continuous confidence score for the
    BTC bull-run condition.

    Works from **any** BTC price level: the rolling-high component scores
    based on whether BTC is making new local highs, independent of an
    absolute dollar threshold.  The absolute ATH component adds a bonus
    when price surpasses a hard level (default $100 K).

    Parameters
    ----------
    ath_target : USD price level for the absolute ATH bonus (Component 1).
                 Pass 0 to disable the absolute check entirely.
    ath_hold_bars : Consecutive bars above ath_target for full ATH score.
    volume_window : Look-back for average-volume baseline (Component 3).
    momentum_window : Look-back for momentum calculation (Component 4).
    rolling_high_window : Look-back bars for the rolling-max comparison
                          (Component 2).  Defaults to 60 (≈ 1 minute at
                          1-tick/sec, or ≈ 1 hour at 1-tick/min).
    rolling_high_hold : Consecutive new-high bars for full rolling score.
    """

    def __init__(
        self,
        ath_target: float        = BTC_ATH_TARGET,
        ath_hold_bars: int       = ATH_HOLD_BARS,
        volume_window: int       = VOLUME_AVG_WINDOW,
        momentum_window: int     = MOMENTUM_WINDOW,
        rolling_high_window: int = ROLLING_HIGH_WINDOW,
        rolling_high_hold: int   = ROLLING_HIGH_HOLD,
    ) -> None:
        self.ath_target          = ath_target
        self.ath_hold_bars       = ath_hold_bars
        self.volume_window       = volume_window
        self.momentum_window     = momentum_window
        self.rolling_high_window = rolling_high_window
        self.rolling_high_hold   = rolling_high_hold

        # Price deque must be large enough for the rolling-high lookback
        _price_maxlen = max(volume_window, momentum_window, rolling_high_window) + 10
        self._prices:  deque[float] = deque(maxlen=_price_maxlen)
        self._volumes: deque[float] = deque(maxlen=volume_window + 5)

        # Consecutive-bar counters
        self._bars_above_ath:      int = 0   # Component 1
        self._bars_at_rolling_high: int = 0  # Component 2

        # Last emitted confidence
        self.last_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, btc_price: float, volume: float = 0.0) -> float:
        """
        Feed the latest BTC price (and optionally volume) and return the
        updated bull-run confidence score in [0, 1].

        Parameters
        ----------
        btc_price : Latest BTC/USD price.
        volume    : Volume for this bar (pass 0 if unavailable; volume
                    scoring is skipped and other components compensate).

        Returns
        -------
        float — confidence in [0, 1].
        """
        self._prices.append(float(btc_price))
        if volume > 0:
            self._volumes.append(float(volume))

        score = 0.0

        # --- Component 1: Absolute ATH breakout + hold (weight 0.20) ---
        # Fires only when price ≥ ath_target (e.g. $100 K).  Provides a
        # strong bonus signal for confirmed macro ATH events.
        if self.ath_target > 0:
            if btc_price >= self.ath_target:
                self._bars_above_ath += 1
            else:
                self._bars_above_ath = 0

            if self._bars_above_ath >= self.ath_hold_bars:
                ath_score = min(1.0, self._bars_above_ath / self.ath_hold_bars)
                score += 0.20 * ath_score
            elif self._bars_above_ath > 0:
                score += 0.20 * (self._bars_above_ath / self.ath_hold_bars)

        # --- Component 2: Rolling-high breakout (weight 0.30) ----------
        # Fires when current price exceeds the rolling max of the previous
        # rolling_high_window bars.  Price-level agnostic — works from
        # $74 K, $50 K, or any other entry price.
        prices_list = list(self._prices)
        if len(prices_list) > self.rolling_high_window:
            lookback    = prices_list[-(self.rolling_high_window + 1):-1]
            rolling_max = max(lookback) if lookback else 0.0
            if rolling_max > 0 and btc_price > rolling_max:
                self._bars_at_rolling_high += 1
            else:
                self._bars_at_rolling_high = 0

            if self._bars_at_rolling_high >= self.rolling_high_hold:
                rh_score = min(1.0, self._bars_at_rolling_high / self.rolling_high_hold)
                score += 0.30 * rh_score
            elif self._bars_at_rolling_high > 0:
                score += 0.30 * (self._bars_at_rolling_high / self.rolling_high_hold)
        # else: not enough history yet — 0 contribution (avoids false signals at startup)

        # --- Component 3: Volume confirmation (weight 0.25) ------------
        if len(self._volumes) >= 2:
            avg_vol = sum(list(self._volumes)[:-1]) / (len(self._volumes) - 1)
            if avg_vol > 0 and self._volumes[-1] > avg_vol:
                vol_ratio = min(self._volumes[-1] / avg_vol, 3.0)   # cap at 3×
                score += 0.25 * min((vol_ratio - 1.0) / 2.0 + 0.5, 1.0)
        elif volume > 0:
            score += 0.125   # single bar, no history — half credit

        # --- Component 4: Momentum (weight 0.25) -----------------------
        momentum = self._momentum()
        if momentum > 0:
            score += 0.25 * min(momentum / 0.05, 1.0)   # full credit at 5 % move

        self.last_confidence = round(min(score, 1.0), 4)
        return self.last_confidence

    def reset(self) -> None:
        """Clear all history and reset the detector."""
        self._prices.clear()
        self._volumes.clear()
        self._bars_above_ath       = 0
        self._bars_at_rolling_high = 0
        self.last_confidence       = 0.0

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
