"""
Alt Pump Detector
=================
Scores each altcoin's readiness for its next pump leg on a 0.0–1.0 scale.

Scoring breakdown (matches problem spec):
  0.20 — Consolidation: tight price range over last 5 bars
  0.30 — Breakout:      latest close above the 5-bar high (resistance break)
  0.20 — Volume spike:  current volume > 20-day avg × MIN_VOLUME_SPIKE (1.5×)
  0.15 — RSI:           RSI between RSI_UNDERBOUGHT (30) and RSI_OVERBOUGHT (80)
  0.15 — Momentum:      positive short-term price momentum

Alt Topping Signal (separate boolean):
  - +15 % gain in 1 week
  - RSI > 80 (overbought)
  - Momentum divergence (price higher but momentum weaker than previous bar)
  - Volume declining vs. recent average

Usage::

    from strategies.signals.alt_pump_detector import AltPumpDetector

    det = AltPumpDetector()
    score = det.update("SOL", price=130.5, volume=4_200_000)
    topping = det.is_topping("SOL")
"""

from __future__ import annotations

from collections import deque
from typing import Optional
from strategies.signals.indicators import rsi as _rsi_fn


# ---------------------------------------------------------------------------
# Constants (mirror config section in problem spec)
# ---------------------------------------------------------------------------

RSI_OVERBOUGHT: int     = 80
RSI_UNDERBOUGHT: int    = 30
MIN_VOLUME_SPIKE: float = 1.5     # current volume must exceed avg × this
CONSOLIDATION_DAYS: int = 5       # bars for tight-range check
VOLUME_AVG_WINDOW: int  = 20      # bars for average-volume baseline
TOPPING_GAIN_THRESHOLD: float = 0.15   # +15 % in TOPPING_WINDOW bars
TOPPING_WINDOW: int     = 7            # bars that represent "1 week"
MOMENTUM_WINDOW: int    = 5


class AltPumpDetector:
    """
    Stateful pump-readiness scorer for multiple altcoins.

    One instance can track all alts simultaneously.  Maintains separate
    rolling price / volume history per asset.

    Parameters
    ----------
    consolidation_bars : Number of bars used for the tight-range test.
    volume_window      : Look-back for average-volume baseline.
    volume_spike_ratio : Volume must exceed avg × this to score the spike.
    rsi_overbought     : RSI ceiling for the RSI component.
    rsi_underbought    : RSI floor for the RSI component.
    """

    def __init__(
        self,
        consolidation_bars: int  = CONSOLIDATION_DAYS,
        volume_window: int       = VOLUME_AVG_WINDOW,
        volume_spike_ratio: float = MIN_VOLUME_SPIKE,
        rsi_overbought: int      = RSI_OVERBOUGHT,
        rsi_underbought: int     = RSI_UNDERBOUGHT,
        momentum_window: int     = MOMENTUM_WINDOW,
        topping_gain: float      = TOPPING_GAIN_THRESHOLD,
        topping_window: int      = TOPPING_WINDOW,
    ) -> None:
        self.consolidation_bars  = consolidation_bars
        self.volume_window       = volume_window
        self.volume_spike_ratio  = volume_spike_ratio
        self.rsi_overbought      = rsi_overbought
        self.rsi_underbought     = rsi_underbought
        self.momentum_window     = momentum_window
        self.topping_gain        = topping_gain
        self.topping_window      = topping_window

        maxlen = max(volume_window, topping_window, momentum_window) + 10
        # per-asset deques — created on first update()
        self._prices:     dict[str, deque[float]] = {}
        self._volumes:    dict[str, deque[float]] = {}
        self._maxlen      = maxlen
        self.last_scores: dict[str, float]        = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        asset: str,
        price: float,
        volume: float = 0.0,
    ) -> float:
        """
        Append one bar for *asset* and return its pump-readiness score [0, 1].

        Parameters
        ----------
        asset  : Ticker string, e.g. "SOL".
        price  : Latest close/last-trade price.
        volume : Bar volume (omit or pass 0 to skip volume scoring).

        Returns
        -------
        float — pump-readiness confidence in [0, 1].
        """
        if asset not in self._prices:
            self._prices[asset]  = deque(maxlen=self._maxlen)
            self._volumes[asset] = deque(maxlen=self._maxlen)

        self._prices[asset].append(float(price))
        if volume > 0:
            self._volumes[asset].append(float(volume))

        score = self._compute_score(asset)
        self.last_scores[asset] = round(score, 4)
        return self.last_scores[asset]

    def is_topping(self, asset: str) -> bool:
        """
        Return True when an alt shows topping signals (time to exit / rotate).

        Criteria (all must fire):
          - Gained >= 15 % in the last TOPPING_WINDOW bars
          - RSI > RSI_OVERBOUGHT (80)
          - Current momentum weaker than previous momentum (divergence)
          - Recent volume declining vs. mid-window average
        """
        prices  = list(self._prices.get(asset, []))
        volumes = list(self._volumes.get(asset, []))

        if len(prices) < self.topping_window + 1:
            return False

        # --- Criterion 1: significant recent gain -----------------------
        base = prices[-(self.topping_window + 1)]
        if base <= 0:
            return False
        gain = (prices[-1] - base) / base
        if gain < self.topping_gain:
            return False

        # --- Criterion 2: RSI overbought --------------------------------
        rsi_val = _rsi_fn(prices)
        if rsi_val is None or rsi_val <= self.rsi_overbought:
            return False

        # --- Criterion 3: momentum divergence ---------------------------
        mom_now  = self._momentum(prices, self.momentum_window)
        mom_prev = self._momentum(prices[:-1], self.momentum_window)
        if mom_now >= mom_prev:
            return False   # momentum still strengthening — not diverging yet

        # --- Criterion 4: volume declining ------------------------------
        if len(volumes) >= 4:
            recent_avg = sum(volumes[-2:]) / 2
            prior_avg  = sum(volumes[-4:-2]) / 2
            if prior_avg > 0 and recent_avg >= prior_avg:
                return False  # volume not declining

        return True

    def reset(self, asset: Optional[str] = None) -> None:
        """Clear history. Pass asset name to reset one asset, or None for all."""
        if asset is None:
            self._prices.clear()
            self._volumes.clear()
            self.last_scores.clear()
        else:
            self._prices.pop(asset, None)
            self._volumes.pop(asset, None)
            self.last_scores.pop(asset, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, asset: str) -> float:
        prices  = list(self._prices[asset])
        volumes = list(self._volumes.get(asset, []))

        score = 0.0

        # --- 0.20 Consolidation: tight range over last N bars ----------
        if len(prices) >= self.consolidation_bars:
            window = prices[-self.consolidation_bars:]
            price_range = (max(window) - min(window)) / max(min(window), 1e-9)
            # A range < 3 % is "tight"; scale from 0 at 10 % to 1 at 0 %
            if price_range < 0.10:
                score += 0.20 * max(0.0, (0.10 - price_range) / 0.10)

        # --- 0.30 Breakout: close > consolidation high -----------------
        if len(prices) >= self.consolidation_bars + 1:
            resistance = max(prices[-(self.consolidation_bars + 1):-1])
            if prices[-1] > resistance:
                breakout_pct = (prices[-1] - resistance) / max(resistance, 1e-9)
                score += 0.30 * min(breakout_pct / 0.05 + 0.5, 1.0)

        # --- 0.20 Volume spike: current > 20d avg × spike_ratio --------
        if len(volumes) >= 2:
            avg_vol = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
            if avg_vol > 0:
                ratio = volumes[-1] / avg_vol
                if ratio >= self.volume_spike_ratio:
                    score += 0.20 * min((ratio - self.volume_spike_ratio + 1.0), 2.0) / 2.0
        elif volumes:
            score += 0.10  # single bar — half credit

        # --- 0.15 RSI: 30 < RSI < 80 -----------------------------------
        rsi_val = _rsi_fn(prices)
        if rsi_val is not None:
            if self.rsi_underbought < rsi_val < self.rsi_overbought:
                # Full credit in mid-range; partial toward the extremes
                rsi_range_ratio = (self.rsi_overbought - rsi_val) / (
                    self.rsi_overbought - self.rsi_underbought
                )
                score += 0.15 * min(rsi_range_ratio * 2, 1.0)

        # --- 0.15 Momentum: positive short-term return -----------------
        mom = self._momentum(prices, self.momentum_window)
        if mom > 0:
            score += 0.15 * min(mom / 0.05, 1.0)  # full at 5 %

        return min(score, 1.0)

    @staticmethod
    def _momentum(prices: list[float], window: int) -> float:
        if len(prices) < window + 1:
            return 0.0
        base = prices[-(window + 1)]
        if base <= 0:
            return 0.0
        return (prices[-1] - base) / base
