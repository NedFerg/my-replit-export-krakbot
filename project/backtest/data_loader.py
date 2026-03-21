"""
Historical data loader — CCXT-based fetcher and CSV manager.

Fetches OHLCV data from Kraken via CCXT and stores it as CSV files in
data/historical/.  Subsequent calls load from the cached CSV if available,
only fetching missing date ranges from the exchange.

Usage
-----
    from project.backtest.data_loader import DataLoader

    loader = DataLoader()
    df = loader.load("BTC/USD", timeframe="1h",
                     start_date="2023-01-01", end_date="2023-12-31")
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    EXCHANGE_ID,
    FETCH_LIMIT,
    FETCH_SLEEP_SECONDS,
)

logger = logging.getLogger(__name__)


def _asset_to_filename(symbol: str, timeframe: str) -> str:
    """Convert 'BTC/USD' + '1h' → 'BTC_USD_1h.csv'."""
    return symbol.replace("/", "_") + f"_{timeframe}.csv"


def _parse_ohlcv(raw: list) -> pd.DataFrame:
    """Convert a CCXT OHLCV list-of-lists to a DataFrame."""
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


class DataLoader:
    """
    Manages loading and caching of historical OHLCV data.

    Parameters
    ----------
    data_dir : Path | None
        Directory to store CSV files.  Defaults to DATA_DIR from config.
    exchange_id : str
        CCXT exchange identifier.  Defaults to "kraken".
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        exchange_id: str = EXCHANGE_ID,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exchange_id = exchange_id
        self._exchange = None   # lazily initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of OHLCV data for *symbol*.

        Loads from the CSV cache if available; fetches from CCXT otherwise.
        Pass ``force_refresh=True`` to always re-fetch from the exchange.

        Parameters
        ----------
        symbol      : e.g. "BTC/USD"
        timeframe   : "1h" | "4h" | "1d"
        start_date  : ISO-8601 string, e.g. "2023-01-01"
        end_date    : ISO-8601 string, e.g. "2023-12-31"
        force_refresh : discard cache and re-fetch

        Returns
        -------
        pd.DataFrame with columns: timestamp, open, high, low, close, volume
        """
        csv_path = self.data_dir / _asset_to_filename(symbol, timeframe)

        if csv_path.exists() and not force_refresh:
            logger.info("Loading %s %s from cache: %s", symbol, timeframe, csv_path)
            df = self._load_csv(csv_path)
        else:
            logger.info("Fetching %s %s from %s …", symbol, timeframe, self.exchange_id)
            df = self._fetch_from_exchange(symbol, timeframe, start_date, end_date)
            self._save_csv(df, csv_path)

        # Slice to requested date range
        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
            df = df[df["timestamp"] < end_ts]

        logger.info(
            "Loaded %d candles for %s %s (%s → %s)",
            len(df),
            symbol,
            timeframe,
            df["timestamp"].iloc[0] if not df.empty else "N/A",
            df["timestamp"].iloc[-1] if not df.empty else "N/A",
        )
        return df.reset_index(drop=True)

    def load_multiple(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        start_date: str | None = None,
        end_date: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.

        Returns
        -------
        dict mapping symbol → DataFrame
        """
        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load(
                    symbol, timeframe, start_date, end_date, force_refresh
                )
            except Exception as exc:
                logger.error("Failed to load %s: %s", symbol, exc)
        return result

    def fetch_and_save(
        self,
        symbols: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str,
    ) -> None:
        """
        Fetch and cache OHLCV data for all symbol × timeframe combinations.

        Intended for use by scripts/fetch_historical_data.py.
        """
        total = len(symbols) * len(timeframes)
        done = 0
        for symbol in symbols:
            for tf in timeframes:
                done += 1
                logger.info("[%d/%d] %s %s …", done, total, symbol, tf)
                try:
                    self.load(symbol, tf, start_date, end_date, force_refresh=False)
                except Exception as exc:
                    logger.error("  Failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_exchange(self):
        """Lazily create and return the CCXT exchange instance."""
        if self._exchange is None:
            try:
                import ccxt
            except ImportError as exc:
                raise ImportError(
                    "ccxt is required for fetching historical data: pip install ccxt"
                ) from exc

            exchange_cls = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_cls({"enableRateLimit": True})
            logger.info("Initialised CCXT exchange: %s", self.exchange_id)
        return self._exchange

    def _fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        """Fetch full date range from CCXT in paginated calls."""
        exchange = self._get_exchange()

        # Convert start_date to millisecond timestamp
        if start_date:
            since_ms = int(
                datetime.strptime(start_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )
        else:
            since_ms = None

        end_ms: int | None = None
        if end_date:
            end_ms = int(
                (
                    datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    .timestamp()
                    + 86400  # include the full end day
                )
                * 1000
            )

        all_candles: list = []
        current_since = since_ms

        while True:
            try:
                raw = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=FETCH_LIMIT,
                )
            except Exception as exc:
                logger.error("CCXT fetch error for %s %s: %s", symbol, timeframe, exc)
                break

            if not raw:
                break

            all_candles.extend(raw)
            last_ts = raw[-1][0]

            # Stop if we've passed the end date or got a partial page
            if (end_ms and last_ts >= end_ms) or len(raw) < FETCH_LIMIT:
                break

            current_since = last_ts + 1  # next page starts after last candle
            time.sleep(FETCH_SLEEP_SECONDS)

        if not all_candles:
            logger.warning("No candles returned for %s %s", symbol, timeframe)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = _parse_ohlcv(all_candles)

        # Trim to requested end date
        if end_ms:
            df = df[df["timestamp"] < pd.Timestamp(end_ms, unit="ms", tz="UTC")]

        # Drop duplicates that may appear at page boundaries
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        return df

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load a cached OHLCV CSV and ensure proper types."""
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        """Save a DataFrame to CSV."""
        if df.empty:
            logger.warning("Not saving empty DataFrame to %s", path)
            return
        df.to_csv(path, index=False)
        logger.info("Saved %d candles → %s", len(df), path)
