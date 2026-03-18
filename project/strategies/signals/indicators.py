"""
Shared technical indicator utilities for signal detectors.
"""

from __future__ import annotations

from typing import Optional


def rsi(prices: list[float], period: int = 14) -> Optional[float]:
    """
    Compute the Relative Strength Index (RSI) over the last *period* bars.

    Parameters
    ----------
    prices : List of price values (chronological order).
    period : Look-back period (default 14).

    Returns
    -------
    float in [0, 100] or None if there are fewer than period+1 data points.
    """
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
