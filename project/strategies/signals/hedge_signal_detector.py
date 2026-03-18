"""
Hedge Signal Detector
=====================
Reads RSI, Bollinger Bands, and resistance/support levels for a tracked asset
and emits a real-time hedge recommendation every bar.

The output drives the ``_apply_hedge_overlay()`` method in BullBearRotationalTrader,
which places small counter-positions so the portfolio can profit regardless of
whether the next move is up or down.

Design
------
Each signal component contributes to one of two scores:

  ``overbought_score`` (0.0–3.0 raw, normalised to 0–1)
      Fires when the market shows signs of a pending breakdown / pullback.
      → Hedge by opening or scaling up a **short ETF** position (ETHD/SETH).

  ``oversold_score`` (0.0–3.0 raw, normalised to 0–1)
      Fires when the market shows signs of a pending breakout / bounce.
      → Hedge by ensuring a **long ETF** floor position (ETHU/SLON/XXRP).

Signal components (each contributes at most 1 point per score):

  1. RSI (weight 1.0 each direction)
       RSI > ``rsi_overbought``   → +1 to overbought_score
       RSI < ``rsi_oversold``     → +1 to oversold_score

  2. Bollinger Bands %B (weight 1.0 each direction)
       %B > 1.0 (above upper band) → +1 to overbought_score  (partial at %B > 0.9)
       %B < 0.0 (below lower band) → +1 to oversold_score    (partial at %B < 0.1)

  3. Resistance / support proximity (weight 1.0 each direction)
       Price at or above rolling N-bar high → +1 to overbought_score
       Price at or below rolling N-bar low  → +1 to oversold_score

Final recommendation
--------------------
  ``bias``        : "long", "short", or "neutral"
  ``short_size``  : fraction of equity to allocate to short ETF hedge (0–``max_short``)
  ``long_size``   : fraction of equity to allocate to long ETF floor   (0–``max_long``)

Usage::

    from strategies.signals.hedge_signal_detector import HedgeSignalDetector

    det = HedgeSignalDetector("BTC")
    rec = det.update(btc_price, volume=btc_vol)
    print(rec.bias, rec.short_size, rec.long_size)
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple, Optional

from strategies.signals.indicators import (
    rsi as _rsi,
    bollinger_bands,
    resistance_support_level,
)


# ---------------------------------------------------------------------------
# Default configuration (override per-instance via constructor)
# ---------------------------------------------------------------------------

RSI_OVERBOUGHT:  int   = 72    # RSI above this → overbought (conservative — slightly lower than 80)
RSI_OVERSOLD:    int   = 35    # RSI below this → oversold   (slightly higher than 30)
BB_WINDOW:       int   = 20    # Bollinger Bands look-back
BB_STD:          float = 2.0   # Bollinger Bands standard deviation multiplier
RS_WINDOW:       int   = 20    # Resistance/support rolling window
RS_TOUCH_PCT:    float = 0.01  # Within 1 % of high/low = "at level"
RSI_PERIOD:      int   = 14    # RSI calculation period

# Position-size bounds for the hedge overlay
HEDGE_SHORT_MAX: float = 0.08  # max 8 % allocation to short ETF
HEDGE_LONG_MAX:  float = 0.10  # max 10 % allocation to long ETF floor

# Minimum number of bars before any hedge is opened (prevents false signals at startup)
MIN_HISTORY_BARS: int = 25


class HedgeRecommendation(NamedTuple):
    """
    Output of HedgeSignalDetector.update().

    Attributes
    ----------
    bias           : Directional bias — "long", "short", or "neutral".
    short_size     : Suggested short-ETF allocation (fraction of portfolio).
                     0.0 = no short hedge needed.
    long_size      : Suggested long-ETF floor allocation (fraction of portfolio).
                     0.0 = no additional long floor needed.
    overbought_score : Normalised [0, 1] overbought intensity.
    oversold_score   : Normalised [0, 1] oversold intensity.
    rsi_val        : Latest RSI reading, or None if insufficient history.
    pct_b          : Latest Bollinger %B, or None if insufficient history.
    at_resistance  : True if price is within touch_pct % of rolling high.
    at_support     : True if price is within touch_pct % of rolling low.
    """
    bias:             str
    short_size:       float
    long_size:        float
    overbought_score: float
    oversold_score:   float
    rsi_val:          Optional[float]
    pct_b:            Optional[float]
    at_resistance:    bool
    at_support:       bool


_NEUTRAL = HedgeRecommendation(
    bias="neutral",
    short_size=0.0,
    long_size=0.0,
    overbought_score=0.0,
    oversold_score=0.0,
    rsi_val=None,
    pct_b=None,
    at_resistance=False,
    at_support=False,
)


class HedgeSignalDetector:
    """
    Stateful real-time hedge signal detector for a single asset.

    Maintains rolling price / volume history and combines RSI, Bollinger
    Bands, and resistance/support proximity into a ``HedgeRecommendation``
    that BullBearRotationalTrader uses to size both long and short hedge
    legs regardless of the current market phase.

    Parameters
    ----------
    asset             : Ticker name for logging (e.g. "BTC", "ETH").
    rsi_overbought    : RSI level above which the asset is considered overbought.
    rsi_oversold      : RSI level below which the asset is considered oversold.
    bb_window         : Bollinger Bands SMA window.
    bb_std            : Bollinger Bands standard deviation multiplier.
    rs_window         : Resistance/support look-back window.
    rs_touch_pct      : Proximity fraction to flag "at resistance/support".
    rsi_period        : RSI calculation period.
    max_short         : Max short-ETF allocation (fraction of equity).
    max_long          : Max long-ETF floor allocation (fraction of equity).
    min_history_bars  : Bars required before any hedge fires.
    """

    def __init__(
        self,
        asset:            str,
        rsi_overbought:   int   = RSI_OVERBOUGHT,
        rsi_oversold:     int   = RSI_OVERSOLD,
        bb_window:        int   = BB_WINDOW,
        bb_std:           float = BB_STD,
        rs_window:        int   = RS_WINDOW,
        rs_touch_pct:     float = RS_TOUCH_PCT,
        rsi_period:       int   = RSI_PERIOD,
        max_short:        float = HEDGE_SHORT_MAX,
        max_long:         float = HEDGE_LONG_MAX,
        min_history_bars: int   = MIN_HISTORY_BARS,
    ) -> None:
        self.asset            = asset
        self.rsi_overbought   = rsi_overbought
        self.rsi_oversold     = rsi_oversold
        self.bb_window        = bb_window
        self.bb_std           = bb_std
        self.rs_window        = rs_window
        self.rs_touch_pct     = rs_touch_pct
        self.rsi_period       = rsi_period
        self.max_short        = max_short
        self.max_long         = max_long
        self.min_history_bars = min_history_bars

        _maxlen = max(bb_window, rs_window, rsi_period) * 2 + 10
        self._prices: deque[float] = deque(maxlen=_maxlen)

        # Last computed recommendation (for callers that poll without new data)
        self.last_recommendation: HedgeRecommendation = _NEUTRAL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, price: float, volume: float = 0.0) -> HedgeRecommendation:
        """
        Append one bar and return the updated hedge recommendation.

        Parameters
        ----------
        price  : Latest traded price for ``self.asset``.
        volume : Bar volume (currently unused but reserved for future signals
                 such as volume-divergence confirmation).

        Returns
        -------
        HedgeRecommendation namedtuple.
        """
        if price <= 0:
            return self.last_recommendation

        self._prices.append(float(price))
        prices = list(self._prices)

        # Not enough history — return neutral to avoid false signals
        if len(prices) < self.min_history_bars:
            return _NEUTRAL

        ob_score = 0.0   # overbought
        os_score = 0.0   # oversold

        # ---- Signal 1: RSI ------------------------------------------
        rsi_val = _rsi(prices, period=self.rsi_period)
        if rsi_val is not None:
            if rsi_val >= self.rsi_overbought:
                # Full credit at overbought; extra credit above 80
                ob_score += min(1.0, (rsi_val - self.rsi_overbought) / (100 - self.rsi_overbought) + 0.5)
            elif rsi_val <= self.rsi_oversold:
                os_score += min(1.0, (self.rsi_oversold - rsi_val) / self.rsi_oversold + 0.5)
        else:
            rsi_val = None

        # ---- Signal 2: Bollinger Bands %B ---------------------------
        bb = bollinger_bands(prices, window=self.bb_window, num_std=self.bb_std)
        pct_b: Optional[float] = None
        if bb is not None:
            pct_b = bb.pct_b
            if pct_b >= 1.0:
                ob_score += 1.0            # above upper band
            elif pct_b >= 0.90:
                ob_score += (pct_b - 0.90) / 0.10   # partial: 0.90–1.0 → 0–1
            elif pct_b <= 0.0:
                os_score += 1.0            # below lower band
            elif pct_b <= 0.10:
                os_score += (0.10 - pct_b) / 0.10   # partial: 0.10–0 → 0–1

        # ---- Signal 3: Resistance / support proximity ---------------
        rs = resistance_support_level(
            prices,
            window=self.rs_window,
            touch_pct=self.rs_touch_pct,
        )
        at_resistance = False
        at_support    = False
        if rs is not None:
            at_resistance = rs.at_resistance
            at_support    = rs.at_support
            if at_resistance:
                ob_score += 1.0
            elif rs.proximity > 0.70:
                # Near (but not touching) resistance: partial signal
                ob_score += (rs.proximity - 0.70) / 0.30
            if at_support:
                os_score += 1.0
            elif rs.proximity < -0.70:
                # Near (but not touching) support: partial signal
                os_score += (-rs.proximity - 0.70) / 0.30

        # ---- Normalise scores (max raw = 3.0) -----------------------
        ob_norm = min(ob_score / 3.0, 1.0)
        os_norm = min(os_score / 3.0, 1.0)

        # ---- Determine bias and hedge sizes -------------------------
        if ob_norm > os_norm + 0.15:
            bias = "short"
        elif os_norm > ob_norm + 0.15:
            bias = "long"
        else:
            bias = "neutral"

        short_size = self.max_short * ob_norm   # 0 → max_short linearly with signal
        long_size  = self.max_long  * os_norm   # 0 → max_long  linearly with signal

        rec = HedgeRecommendation(
            bias=bias,
            short_size=round(short_size, 4),
            long_size=round(long_size, 4),
            overbought_score=round(ob_norm, 4),
            oversold_score=round(os_norm, 4),
            rsi_val=round(rsi_val, 2) if rsi_val is not None else None,
            pct_b=round(pct_b, 4) if pct_b is not None else None,
            at_resistance=at_resistance,
            at_support=at_support,
        )
        self.last_recommendation = rec
        return rec

    def reset(self) -> None:
        """Clear all history and reset to neutral."""
        self._prices.clear()
        self.last_recommendation = _NEUTRAL
