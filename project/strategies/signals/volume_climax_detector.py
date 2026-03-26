"""
Volume Climax Detector
======================
Detects two related but distinct high-volume reversal events in price/volume
data:

  **Volume Capitulation** — panic selling (or buying) characterized by:
    • A sharp price move against the prior trend.
    • An extreme volume spike (sellers/buyers flooding the market).
    • RSI at or near oversold (for sell-side capitulation) or overbought
      (for buy-side capitulation) levels.

  **Volume Exhaustion** — the trend running out of steam, characterized by:
    • Continued high or spiking volume.
    • Narrowing price range per unit of volume (little net movement despite
      heavy activity).
    • Volume beginning to decline after the climax peak, while price
      stagnates.

These events frequently coincide with meaningful price reversals and are
therefore useful input signals for stop-loss tightening, position reduction,
or counter-trend entry logic.

Scoring
-------
Both signals are returned as continuous scores in [0, 1]:

  ``capitulation_score`` — weighted combination of:
    Component 1  Volume spike                  (weight 0.35)
    Component 2  Sharp adverse price move       (weight 0.35)
    Component 3  RSI at extreme                 (weight 0.30)

  ``exhaustion_score`` — weighted combination of:
    Component 1  Volume spike                  (weight 0.40)
    Component 2  Narrow price range vs. volume (weight 0.35)
    Component 3  Volume declining after peak   (weight 0.25)

Boolean flags ``is_capitulation`` / ``is_exhaustion`` fire when the
corresponding score exceeds a configurable threshold (default 0.60).

Usage::

    from strategies.signals.volume_climax_detector import VolumeClimaxDetector

    det = VolumeClimaxDetector()
    result = det.update(price=45_000, volume=18_500, price_range=1_200)
    print(result.capitulation_score, result.is_capitulation)
    print(result.exhaustion_score,   result.is_exhaustion)
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple, Optional

from strategies.signals.indicators import rsi as _rsi_fn


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

VOLUME_AVG_WINDOW: int         = 20      # bars used to compute the baseline volume average
VOLUME_SPIKE_RATIO: float      = 2.5     # volume must be ≥ avg × this for a "climax" spike
PRICE_CHANGE_WINDOW: int       = 5       # bars over which the adverse price move is measured
PRICE_DECLINE_THRESHOLD: float = 0.03   # 3 % adverse move to score max on Component 2
RSI_PERIOD: int                = 14      # look-back for RSI calculation
RSI_OVERSOLD: int              = 30      # RSI below this → capitulation (sell-side)
RSI_OVERBOUGHT: int            = 70      # RSI above this → capitulation (buy-side)
EXHAUSTION_RANGE_WINDOW: int   = 3       # bars to average for narrow-range detection
VOLUME_DECLINE_BARS: int       = 3       # bars to check for declining volume after peak
CAPITULATION_THRESHOLD: float  = 0.60   # score above which is_capitulation fires
EXHAUSTION_THRESHOLD: float    = 0.60   # score above which is_exhaustion fires
MIN_HISTORY_BARS: int          = 22      # minimum bars before any signal fires


class ClimaxResult(NamedTuple):
    """
    Output of :meth:`VolumeClimaxDetector.update`.

    Attributes
    ----------
    capitulation_score : Continuous confidence that a volume capitulation event
                         is occurring (0.0 = no signal, 1.0 = maximum confidence).
    exhaustion_score   : Continuous confidence that volume exhaustion is
                         occurring (0.0 = no signal, 1.0 = maximum confidence).
    is_capitulation    : True when ``capitulation_score`` ≥ the detector's
                         ``capitulation_threshold``.
    is_exhaustion      : True when ``exhaustion_score`` ≥ the detector's
                         ``exhaustion_threshold``.
    volume_ratio       : Current bar volume divided by the rolling average
                         (``float('nan')`` if no average yet).
    rsi_val            : Latest RSI reading, or ``None`` if insufficient history.
    price_change       : N-bar price return (negative = price fell).
    """
    capitulation_score: float
    exhaustion_score:   float
    is_capitulation:    bool
    is_exhaustion:      bool
    volume_ratio:       float
    rsi_val:            Optional[float]
    price_change:       float


_NEUTRAL = ClimaxResult(
    capitulation_score=0.0,
    exhaustion_score=0.0,
    is_capitulation=False,
    is_exhaustion=False,
    volume_ratio=float("nan"),
    rsi_val=None,
    price_change=0.0,
)


class VolumeClimaxDetector:
    """
    Stateful detector that emits volume capitulation and exhaustion scores.

    A single instance tracks one price series.  Call :meth:`update` once per
    bar (or tick) to feed new data and receive updated :class:`ClimaxResult`
    readings.

    Parameters
    ----------
    volume_avg_window        : Bars used to compute the rolling average volume
                               baseline.
    volume_spike_ratio       : Minimum ratio of current volume to rolling
                               average to qualify as a "climax" volume spike.
    price_change_window      : Bars over which the adverse price move is
                               measured for the capitulation score.
    price_decline_threshold  : Fractional price move that earns the maximum
                               Component-2 capitulation score (default 3 %).
    rsi_period               : RSI calculation look-back period.
    rsi_oversold             : RSI below this threshold scores as oversold
                               (sell-side capitulation).
    rsi_overbought           : RSI above this threshold scores as overbought
                               (buy-side capitulation).
    exhaustion_range_window  : Bars over which the average intra-bar price
                               range is computed for exhaustion scoring.
    volume_decline_bars      : Bars checked for declining volume after the
                               climax peak.
    capitulation_threshold   : ``capitulation_score`` must exceed this for
                               ``is_capitulation`` to be True (default 0.60).
    exhaustion_threshold     : ``exhaustion_score`` must exceed this for
                               ``is_exhaustion`` to be True (default 0.60).
    min_history_bars         : Minimum bars of history required before any
                               signal is emitted.
    """

    def __init__(
        self,
        volume_avg_window:       int   = VOLUME_AVG_WINDOW,
        volume_spike_ratio:      float = VOLUME_SPIKE_RATIO,
        price_change_window:     int   = PRICE_CHANGE_WINDOW,
        price_decline_threshold: float = PRICE_DECLINE_THRESHOLD,
        rsi_period:              int   = RSI_PERIOD,
        rsi_oversold:            int   = RSI_OVERSOLD,
        rsi_overbought:          int   = RSI_OVERBOUGHT,
        exhaustion_range_window: int   = EXHAUSTION_RANGE_WINDOW,
        volume_decline_bars:     int   = VOLUME_DECLINE_BARS,
        capitulation_threshold:  float = CAPITULATION_THRESHOLD,
        exhaustion_threshold:    float = EXHAUSTION_THRESHOLD,
        min_history_bars:        int   = MIN_HISTORY_BARS,
    ) -> None:
        self.volume_avg_window       = volume_avg_window
        self.volume_spike_ratio      = volume_spike_ratio
        self.price_change_window     = price_change_window
        self.price_decline_threshold = price_decline_threshold
        self.rsi_period              = rsi_period
        self.rsi_oversold            = rsi_oversold
        self.rsi_overbought          = rsi_overbought
        self.exhaustion_range_window = exhaustion_range_window
        self.volume_decline_bars     = volume_decline_bars
        self.capitulation_threshold  = capitulation_threshold
        self.exhaustion_threshold    = exhaustion_threshold
        self.min_history_bars        = min_history_bars

        _maxlen = max(
            volume_avg_window,
            price_change_window,
            rsi_period,
            exhaustion_range_window,
            volume_decline_bars,
        ) * 2 + 10

        self._prices:  deque[float] = deque(maxlen=_maxlen)
        self._volumes: deque[float] = deque(maxlen=_maxlen)
        self._ranges:  deque[float] = deque(maxlen=_maxlen)  # intra-bar high−low range

        # Last emitted result (for callers that poll without new data)
        self.last_result: ClimaxResult = _NEUTRAL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        price:       float,
        volume:      float = 0.0,
        price_range: float = 0.0,
    ) -> ClimaxResult:
        """
        Append one bar and return the updated :class:`ClimaxResult`.

        Parameters
        ----------
        price       : Closing (or latest traded) price for this bar.
        volume      : Total volume traded during this bar.
                      Pass ``0`` if volume data is unavailable; volume-based
                      components will score 0 until real data is provided.
        price_range : Intra-bar high − low range.  Used by the exhaustion
                      narrow-range component.  When ``0`` the detector
                      estimates range as the absolute difference between the
                      current and previous close.

        Returns
        -------
        :class:`ClimaxResult`
        """
        if price <= 0:
            return self.last_result

        self._prices.append(float(price))

        if volume > 0:
            self._volumes.append(float(volume))

        # Estimate intra-bar range if not supplied
        effective_range = price_range
        if effective_range <= 0 and len(self._prices) >= 2:
            effective_range = abs(self._prices[-1] - self._prices[-2])
        self._ranges.append(float(effective_range))

        # Need enough history to avoid false signals at startup
        if len(self._prices) < self.min_history_bars:
            return _NEUTRAL

        prices  = list(self._prices)
        volumes = list(self._volumes)
        ranges  = list(self._ranges)

        # ---- Shared: volume ratio ----------------------------------------
        volume_ratio = self._volume_ratio(volumes)

        # ---- RSI (shared across both scores) -----------------------------
        rsi_val = _rsi_fn(prices, period=self.rsi_period)

        # ---- Capitulation score -----------------------------------------
        cap_score = self._capitulation_score(prices, volumes, volume_ratio, rsi_val)

        # ---- Exhaustion score -------------------------------------------
        exh_score = self._exhaustion_score(volumes, ranges, volume_ratio)

        # ---- N-bar price change (informational) -------------------------
        n = self.price_change_window
        if len(prices) >= n + 1:
            base = prices[-(n + 1)]
            price_change = (prices[-1] - base) / base if base > 0 else 0.0
        else:
            price_change = 0.0

        result = ClimaxResult(
            capitulation_score=round(min(cap_score, 1.0), 4),
            exhaustion_score=round(min(exh_score, 1.0), 4),
            is_capitulation=cap_score >= self.capitulation_threshold,
            is_exhaustion=exh_score >= self.exhaustion_threshold,
            volume_ratio=round(volume_ratio, 4) if not _is_nan(volume_ratio) else float("nan"),
            rsi_val=round(rsi_val, 2) if rsi_val is not None else None,
            price_change=round(price_change, 6),
        )
        self.last_result = result
        return result

    def reset(self) -> None:
        """Clear all history and reset to neutral."""
        self._prices.clear()
        self._volumes.clear()
        self._ranges.clear()
        self.last_result = _NEUTRAL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _volume_ratio(self, volumes: list[float]) -> float:
        """Return current volume / rolling-average volume, or NaN if unavailable."""
        if len(volumes) < 2:
            return float("nan")
        avg = sum(volumes[:-1]) / (len(volumes) - 1)
        if avg <= 0:
            return float("nan")
        return volumes[-1] / avg

    def _capitulation_score(
        self,
        prices:       list[float],
        volumes:      list[float],
        volume_ratio: float,
        rsi_val:      Optional[float],
    ) -> float:
        """
        Compute the capitulation score [0, 1].

        Component 1 — Volume spike (weight 0.35)
            Full credit when volume ≥ spike_ratio × average.
            Partial credit as volume ratio rises from 1× toward spike_ratio.

        Component 2 — Sharp adverse price move (weight 0.35)
            Scores the magnitude of the N-bar price decline (or sharp rise
            for buy-side capitulation).  Full credit at decline ≥
            ``price_decline_threshold``.

        Component 3 — RSI at extreme (weight 0.30)
            Full credit when RSI is below ``rsi_oversold`` (sell-side) or
            above ``rsi_overbought`` (buy-side).  Partial credit between
            those levels.
        """
        score = 0.0

        # Component 1: Volume spike
        if not _is_nan(volume_ratio) and volume_ratio >= 1.0:
            spike_ratio = self.volume_spike_ratio
            vol_contrib = min((volume_ratio - 1.0) / max(spike_ratio - 1.0, 1e-9), 1.0)
            score += 0.35 * vol_contrib

        # Component 2: Sharp adverse price move
        n = self.price_change_window
        if len(prices) >= n + 1:
            base = prices[-(n + 1)]
            if base > 0:
                change = (prices[-1] - base) / base
                # Adverse move: either a sharp decline OR a sharp surge
                # (buy-side capitulation = everyone jumping in at once)
                move_magnitude = abs(change)
                move_contrib = min(move_magnitude / max(self.price_decline_threshold, 1e-9), 1.0)
                score += 0.35 * move_contrib

        # Component 3: RSI at extreme
        if rsi_val is not None:
            if rsi_val <= self.rsi_oversold:
                # Deeper oversold → higher contribution
                rsi_contrib = min(
                    1.0,
                    (self.rsi_oversold - rsi_val) / max(self.rsi_oversold, 1.0) + 0.5,
                )
                score += 0.30 * rsi_contrib
            elif rsi_val >= self.rsi_overbought:
                rsi_contrib = min(
                    1.0,
                    (rsi_val - self.rsi_overbought) / max(100 - self.rsi_overbought, 1.0) + 0.5,
                )
                score += 0.30 * rsi_contrib

        return score

    def _exhaustion_score(
        self,
        volumes: list[float],
        ranges:  list[float],
        volume_ratio: float,
    ) -> float:
        """
        Compute the exhaustion score [0, 1].

        Component 1 — Volume spike (weight 0.40)
            Same volume-spike logic as capitulation Component 1.

        Component 2 — Narrow price range per unit of volume (weight 0.35)
            Compares the current intra-bar range to the rolling average range.
            A high-volume bar with a *below-average* price range signals that
            the large volume is not driving price — buyers and sellers are
            absorbing each other, a classic exhaustion sign.

        Component 3 — Volume declining after peak (weight 0.25)
            Volume has already peaked and is now declining over the last
            ``volume_decline_bars`` bars.
        """
        score = 0.0

        # Component 1: Volume spike
        if not _is_nan(volume_ratio) and volume_ratio >= 1.0:
            spike_ratio = self.volume_spike_ratio
            vol_contrib = min((volume_ratio - 1.0) / max(spike_ratio - 1.0, 1e-9), 1.0)
            score += 0.40 * vol_contrib

        # Component 2: Narrow price range vs. volume (high vol, small range)
        w = self.exhaustion_range_window
        if len(ranges) >= w + 1:
            avg_range = sum(ranges[-(w + 1):-1]) / w
            current_range = ranges[-1]
            if avg_range > 0:
                range_ratio = current_range / avg_range
                # range_ratio < 1 → current range is narrower than average
                # Full exhaustion credit when range_ratio → 0
                if range_ratio < 1.0:
                    range_contrib = min(1.0 - range_ratio, 1.0)
                    score += 0.35 * range_contrib

        # Component 3: Volume declining after peak
        d = self.volume_decline_bars
        if len(volumes) >= d + 1:
            # Check if each successive volume bar is lower than the one before
            recent = volumes[-(d + 1):]
            declining_count = sum(
                1 for i in range(1, len(recent)) if recent[i] < recent[i - 1]
            )
            decline_contrib = declining_count / max(d, 1)
            score += 0.25 * decline_contrib

        return score


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _is_nan(value: float) -> bool:
    """Return True if *value* is NaN (avoids importing math at module level)."""
    return value != value
