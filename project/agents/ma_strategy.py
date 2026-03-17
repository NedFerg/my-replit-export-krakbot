"""
Dual moving-average crossover strategy for Krakbot.

Signal logic (evaluated every tick, per asset):
  short_ma > long_ma  →  target_exposure = +LONG_TARGET  (long)
  short_ma ≤ long_ma  →  target_exposure =  0.0          (flat)

The strategy is STATE-based, not event-based:
  • It stays long for the full duration the short MA is above the long MA.
  • It stays flat (no short-selling) while the short MA is below the long MA.
  • Crossover events are logged once per state-change so the console stays
    readable without being flooded every tick.

Warming-up:
  The strategy tracks how many price bars it has accumulated per asset.
  self.ready is False until every asset has at least `long_window` bars.
  Callers should check self.ready before calling compute_signals().

Usage (inside agent.step()):
    self.ma_strategy.update(live_prices)        # feed new bar
    if not self.ma_strategy.ready:
        ...                                     # still warming up
    signals = self.ma_strategy.compute_signals(agent)
    # agent.target_exposures is now populated
"""

import collections
from typing import Dict, List


class MAStrategy:
    """
    Simple dual-MA crossover strategy.

    Parameters
    ----------
    assets       : list of asset names matching agent.assets
    short_window : short moving-average lookback (default 5 bars)
    long_window  : long  moving-average lookback (default 20 bars)
    long_target  : target exposure fraction when bullish (default 0.02 = 2 %)
    """

    def __init__(
        self,
        assets: List[str],
        short_window: int = 5,
        long_window:  int = 20,
        long_target:  float = 0.02,
    ):
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be < long_window ({long_window})"
            )

        self.assets       = assets
        self.short_window = short_window
        self.long_window  = long_window
        self.long_target  = long_target

        self._history: Dict[str, collections.deque] = {
            a: collections.deque(maxlen=long_window + 1) for a in assets
        }

        self._prev_signal: Dict[str, float] = {a: 0.0 for a in assets}

        print(
            f"[MA STRATEGY] Initialized  short={short_window}  long={long_window}  "
            f"target={long_target*100:.1f}%  assets={assets}\n"
            f"              Warming up — need {long_window} bars per asset."
        )

    def update(self, prices: dict) -> None:
        """Append the latest prices to each asset's rolling history."""
        for a in self.assets:
            p = prices.get(a)
            if p and float(p) > 0:
                self._history[a].append(float(p))

    @property
    def ready(self) -> bool:
        """True once every asset has accumulated at least `long_window` bars."""
        return all(len(h) >= self.long_window for h in self._history.values())

    @property
    def bars_collected(self) -> Dict[str, int]:
        """Current bar count per asset."""
        return {a: len(self._history[a]) for a in self.assets}

    def _sma(self, asset: str, window: int) -> float | None:
        """Simple moving average over the last `window` bars, or None if not enough data."""
        h = self._history[asset]
        if len(h) < window:
            return None
        tail = list(h)[-window:]
        return sum(tail) / len(tail)

    def compute_signals(self, agent) -> Dict[str, float]:
        """
        Compute MA crossover signals for all assets and write them to
        agent.target_exposures.

        Returns the signals dict  {asset: float}  for logging / inspection.
        Every value is either self.long_target or 0.0.

        Side-effects
        ------------
        - Sets agent.target_exposures[a] for each asset.
        - Sets agent.target_exposure (legacy SOL scalar).
        - Logs a message whenever a crossover is detected (once per state change).
        """
        signals: Dict[str, float] = {}

        live_prices = (
            getattr(agent.broker, "live_prices", {})
            if agent.broker is not None
            else {}
        )

        for a in self.assets:
            short_ma = self._sma(a, self.short_window)
            long_ma  = self._sma(a, self.long_window)

            if short_ma is None or long_ma is None:
                signals[a] = 0.0
            elif short_ma > long_ma:
                signals[a] = self.long_target
            else:
                signals[a] = 0.0

            prev = self._prev_signal.get(a, 0.0)
            if signals[a] != prev:
                direction = (
                    "CROSS UP   → LONG"
                    if signals[a] > 0
                    else "CROSS DOWN → FLAT"
                )
                price = live_prices.get(a, 0.0)
                print(
                    f"[MA STRATEGY] {a:<6} {direction}  "
                    f"price={price:.4f}  "
                    f"short_ma={short_ma:.4f}  "
                    f"long_ma={long_ma:.4f}"
                )

            self._prev_signal[a] = signals[a]
            agent.target_exposures[a] = signals[a]

        agent.target_exposure = agent.target_exposures.get("SOL", 0.0)
        return signals

    def status_line(self) -> str:
        """One-liner for heartbeat / log output."""
        bars   = {a: len(self._history[a]) for a in self.assets}
        longs  = [a for a, s in self._prev_signal.items() if s > 0]
        return (
            f"ready={self.ready}  bars={bars}  "
            f"long={longs or 'none'}  "
            f"short_w={self.short_window}  long_w={self.long_window}"
        )
