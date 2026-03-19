"""
HistoricalDataFeed — candle-by-candle OHLCV iterator.

Wraps a pandas DataFrame of OHLCV data and yields one candle at a time,
matching the structure of the live Kraken data so bot logic does not
need to change between live and backtest modes.

Usage
-----
    feed = HistoricalDataFeed(df, start_date="2023-01-01", end_date="2023-06-30")
    feed.reset()
    while feed.has_more_data():
        candle = feed.get_next_candle()
        # candle keys: timestamp, open, high, low, close, volume
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns (lowercase) after normalisation
_REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, typed copy of *df* ready for the feed."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"HistoricalDataFeed: missing columns {missing}")

    # Parse timestamp → UTC-aware datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


class HistoricalDataFeed:
    """
    Iterates a DataFrame of OHLCV candles one at a time.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with at least the columns:
        timestamp, open, high, low, close, volume.
    start_date : str | None
        ISO-8601 date string (e.g. "2023-01-01").  Candles before this date
        are excluded.  ``None`` → use the earliest available candle.
    end_date : str | None
        ISO-8601 date string (e.g. "2023-06-30").  Candles after this date
        are excluded.  ``None`` → use the latest available candle.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        df = _normalise_df(df)

        if start_date is not None:
            start_ts = pd.Timestamp(start_date, tz="UTC")
            df = df[df["timestamp"] >= start_ts]
        if end_date is not None:
            # Include the whole end day
            end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            df = df[df["timestamp"] < end_ts]

        if df.empty:
            logger.warning("HistoricalDataFeed: no candles in the requested date range.")

        self._df: pd.DataFrame = df.reset_index(drop=True)
        self._index: int = 0

        logger.info(
            "HistoricalDataFeed: %d candles  (%s → %s)",
            len(self._df),
            self._df["timestamp"].iloc[0] if not self._df.empty else "N/A",
            self._df["timestamp"].iloc[-1] if not self._df.empty else "N/A",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Rewind the feed to the first candle."""
        self._index = 0

    def has_more_data(self) -> bool:
        """Return ``True`` if there is at least one more candle to yield."""
        return self._index < len(self._df)

    def get_next_candle(self) -> dict:
        """
        Return the next candle as a dict and advance the internal pointer.

        Keys
        ----
        timestamp : pd.Timestamp  — UTC-aware
        open      : float
        high      : float
        low       : float
        close     : float
        volume    : float
        index     : int           — position in the feed (0-based)

        Raises
        ------
        StopIteration — if no more candles are available.
        """
        if not self.has_more_data():
            raise StopIteration("No more candles in HistoricalDataFeed.")

        row = self._df.iloc[self._index]
        self._index += 1

        return {
            "timestamp": row["timestamp"],
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    float(row["volume"]),
            "index":     self._index - 1,
        }

    def peek_next_candle(self) -> dict | None:
        """Return the next candle without advancing the pointer, or None."""
        if not self.has_more_data():
            return None
        row = self._df.iloc[self._index]
        return {
            "timestamp": row["timestamp"],
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    float(row["volume"]),
            "index":     self._index,
        }

    def remaining(self) -> int:
        """Number of candles not yet consumed."""
        return max(0, len(self._df) - self._index)

    def total(self) -> int:
        """Total number of candles in the feed."""
        return len(self._df)

    def progress(self) -> float:
        """Fraction of candles consumed (0.0 → 1.0)."""
        if len(self._df) == 0:
            return 1.0
        return self._index / len(self._df)

    def as_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying (sliced) DataFrame."""
        return self._df.copy()

    # ------------------------------------------------------------------
    # Iterator protocol — allows ``for candle in feed:``
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        self.reset()
        return self

    def __next__(self) -> dict:
        if not self.has_more_data():
            raise StopIteration
        return self.get_next_candle()

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f"HistoricalDataFeed(total={self.total()}, "
            f"remaining={self.remaining()}, "
            f"progress={self.progress():.1%})"
        )
