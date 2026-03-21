"""
Shared technical indicator utilities for signal detectors.
"""

from __future__ import annotations

import math
from typing import Optional, NamedTuple


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


class BollingerBands(NamedTuple):
    """Result of a Bollinger Bands calculation."""
    upper:  float   # upper band (middle + num_std × std)
    middle: float   # simple moving average over the window
    lower:  float   # lower band (middle − num_std × std)
    std:    float   # standard deviation of the window
    pct_b:  float   # % B: 0 = at lower, 0.5 = at middle, 1 = at upper


def bollinger_bands(
    prices: list[float],
    window: int = 20,
    num_std: float = 2.0,
) -> Optional[BollingerBands]:
    """
    Compute Bollinger Bands for the last *window* bars.

    Parameters
    ----------
    prices  : List of price values (chronological order).
    window  : Look-back period for the SMA (default 20).
    num_std : Number of standard deviations for the band width (default 2).

    Returns
    -------
    BollingerBands namedtuple or None if not enough data.

    Notes
    -----
    %B = (price − lower) / (upper − lower).
      %B > 1.0  → price above upper band (overbought signal)
      %B < 0.0  → price below lower band (oversold signal)
      %B ≈ 0.5  → price at the middle band (neutral)
    """
    if len(prices) < window:
        return None

    window_prices = prices[-window:]
    middle = sum(window_prices) / window
    variance = sum((p - middle) ** 2 for p in window_prices) / window
    std = math.sqrt(variance)
    upper = middle + num_std * std
    lower = middle - num_std * std

    band_width = upper - lower
    if band_width <= 0:
        pct_b = 0.5
    else:
        pct_b = (prices[-1] - lower) / band_width

    return BollingerBands(
        upper=upper,
        middle=middle,
        lower=lower,
        std=std,
        pct_b=pct_b,
    )


class ResistanceSupport(NamedTuple):
    """Proximity of current price to key N-bar resistance and support levels."""
    resistance: float   # rolling N-bar high (resistance level)
    support:    float   # rolling N-bar low (support level)
    # Signed distance: +1 = at resistance, −1 = at support, 0 = mid-range
    proximity:  float
    # True when price is within touch_pct % of resistance
    at_resistance: bool
    # True when price is within touch_pct % of support
    at_support:    bool


def resistance_support_level(
    prices: list[float],
    window: int = 20,
    touch_pct: float = 0.01,
) -> Optional[ResistanceSupport]:
    """
    Identify rolling resistance (N-bar high) and support (N-bar low).

    Parameters
    ----------
    prices    : List of price values (chronological order).
    window    : Look-back window for high/low (default 20).
    touch_pct : Proximity threshold — price within this fraction of the
                resistance/support level triggers ``at_resistance`` /
                ``at_support`` flags (default 1 %).

    Returns
    -------
    ResistanceSupport namedtuple or None if not enough data.

    Notes
    -----
    ``proximity`` is computed as:
        (price − support) / (resistance − support) × 2 − 1
    so it ranges from −1 (at support) through 0 (mid-range) to +1 (at resistance).
    """
    if len(prices) < window + 1:
        return None

    # Use the window *excluding* the current bar to avoid look-ahead
    lookback    = prices[-(window + 1):-1]
    resistance  = max(lookback)
    support     = min(lookback)
    current     = prices[-1]
    price_range = resistance - support

    if price_range <= 0:
        proximity = 0.0
    else:
        proximity = (current - support) / price_range * 2.0 - 1.0

    at_resistance = resistance > 0 and abs(current - resistance) / resistance <= touch_pct
    at_support    = support    > 0 and abs(current - support)    / support    <= touch_pct

    return ResistanceSupport(
        resistance=resistance,
        support=support,
        proximity=proximity,
        at_resistance=at_resistance,
        at_support=at_support,
    )
