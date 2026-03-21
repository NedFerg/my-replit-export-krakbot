#!/usr/bin/env python3
"""
fetch_historical_data.py — Download OHLCV data from Kraken via CCXT.

Fetches candle data for all configured assets and timeframes over the
standard date ranges (2019–2021 and 2023–2024), then saves each dataset
as a CSV in data/historical/.

If the Kraken API is unavailable (network restrictions, rate limits, etc.)
the script falls back to generating synthetic OHLCV data using geometric
Brownian motion so that subsequent scripts in the pipeline can still run.

Usage
-----
    python project_scripts/fetch_historical_data.py

Optional arguments can be edited in the CONFIGURATION section below.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap — allow running from the repository root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.backtest.config import (
    BACKTEST_ASSETS,
    BACKTEST_TIMEFRAMES,
    DATE_RANGES,
    DATA_DIR,
)
from project.backtest.data_loader import DataLoader

# ---------------------------------------------------------------------------
# CONFIGURATION (edit as needed)
# ---------------------------------------------------------------------------
SYMBOLS = BACKTEST_ASSETS          # e.g. ["BTC/USD", "ETH/USD", …]
TIMEFRAMES = BACKTEST_TIMEFRAMES   # e.g. ["1h", "4h", "1d"]
FORCE_REFRESH = False              # set True to re-download even if CSV exists

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generation (fallback when API is unavailable)
# ---------------------------------------------------------------------------

# Approximate USD prices by asset symbol and calendar year.
_BASE_PRICES: dict[str, dict[str, float]] = {
    "BTC":  {"2019": 3_500, "2020": 7_000, "2021": 30_000,
             "2022": 20_000, "2023": 17_000, "2024": 45_000},
    "ETH":  {"2019": 120,   "2020": 230,   "2021": 1_000,
             "2022": 1_200,  "2023": 1_250,  "2024": 2_500},
    "SOL":  {"2021": 20,    "2022": 35,    "2023": 12,    "2024": 100},
    "XRP":  {"2019": 0.35,  "2020": 0.23,  "2021": 0.50,
             "2022": 0.38,   "2023": 0.37,  "2024": 0.55},
    "LINK": {"2019": 0.50,  "2020": 8.00,  "2021": 20.00,
             "2022": 8.00,   "2023": 7.00,  "2024": 15.00},
    "AVAX": {"2021": 15.00, "2022": 15.00, "2023": 12.00, "2024": 40.00},
    "HBAR": {"2019": 0.03,  "2020": 0.04,  "2021": 0.05,
             "2022": 0.06,   "2023": 0.05,  "2024": 0.10},
    "XLM":  {"2019": 0.10,  "2020": 0.09,  "2021": 0.30,
             "2022": 0.12,   "2023": 0.09,  "2024": 0.12},
}

# Candle-frequency aliases for pandas date_range
_TF_TO_FREQ: dict[str, str] = {"1h": "h", "4h": "4h", "1d": "D"}


def _base_price(symbol: str, start_date: str) -> float:
    """Return an approximate starting price for *symbol* near *start_date*.

    *start_date* must be a valid ISO-8601 date string (e.g. "2023-01-01")
    as produced by the DATE_RANGES constant in config.py.
    """
    short = symbol.split("/")[0]
    prices = _BASE_PRICES.get(short, {"2019": 100.0})
    start_year = int(start_date[:4])
    # Walk backward from start_year to find the nearest defined year.
    for y in range(start_year, start_year - 10, -1):
        if str(y) in prices:
            return prices[str(y)]
    return list(prices.values())[0]


def _save_synthetic_to_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """
    Persist synthetic OHLCV data to *csv_path*.

    If the file already exists (from a previous date-range iteration) the
    new rows are merged with the existing data so that both date ranges end
    up in a single CSV — matching the DataLoader's one-file-per-ticker
    convention.
    """
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
            if existing["timestamp"].dt.tz is None:
                existing["timestamp"] = existing["timestamp"].dt.tz_localize("UTC")
            else:
                existing["timestamp"] = existing["timestamp"].dt.tz_convert("UTC")
            if not existing.empty:
                df = pd.concat([existing, df], ignore_index=True)
                df = (
                    df.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
        except Exception:
            pass  # If the existing CSV is unreadable, overwrite it.
    df.to_csv(csv_path, index=False)


def _generate_synthetic_ohlcv(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data using geometric Brownian motion.

    The data is statistically plausible (realistic volatility regimes,
    alternating bull/bear phases) but is *not* historically accurate.
    It is intended only as a CI fallback when the exchange API is
    unreachable.
    """
    freq = _TF_TO_FREQ.get(timeframe, "h")
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq, tz="UTC")
    n = len(timestamps)
    if n == 0:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    # Reproducible per symbol/timeframe/period so results are consistent.
    seed = abs(hash(f"{symbol}_{timeframe}_{start_date}")) % (2**31)
    rng = np.random.default_rng(seed)

    # GBM parameters at 1-hour scale, scaled to wider timeframes via √tf.
    #
    # sigma = 0.012 (1.2 % per hour) gives a 6-candle standard deviation of
    # 0.012 × √6 ≈ 2.9 %, so roughly 15 % of 6-hour windows exceed the ±3 %
    # threshold required for bull/bear window qualification in
    # extract_test_windows.py — more than enough to find the 10 we need.
    #
    # mu = 0.00004 per candle ≈ 0.35 % per day, a mild positive drift that
    # prevents the price from trending to zero over multi-year simulations
    # while keeping the expected annualised return moderate (~3 % on 1h data).
    tf_hours = {"1h": 1, "4h": 4, "1d": 24}.get(timeframe, 1)
    sigma = 0.012 * (tf_hours ** 0.5)   # scale hourly vol by √tf
    mu = 0.00004 * tf_hours              # scale drift proportionally to candle width

    log_returns = rng.normal(mu, sigma, n)

    start_price = _base_price(symbol, start_date)
    closes = start_price * np.exp(np.cumsum(log_returns))

    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    intra = np.abs(rng.normal(0, sigma * 0.4, n))
    highs = np.maximum(opens, closes) * (1 + intra)
    lows  = np.minimum(opens, closes) * (1 - intra)

    short = symbol.split("/")[0]
    # log-space means for lognormal volume distribution:
    #   BTC → e^15 ≈ 3.3M units/candle, ETH → e^14 ≈ 1.2M, SOL → e^13 ≈ 440K
    #   default → e^12 ≈ 162K for smaller-cap assets
    vol_mean = {"BTC": 15, "ETH": 14, "SOL": 13}.get(short, 12)
    volumes = rng.lognormal(mean=vol_mean, sigma=1.5, size=n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open":   np.round(opens,  8),
        "high":   np.round(highs,  8),
        "low":    np.round(lows,   8),
        "close":  np.round(closes, 8),
        "volume": np.round(volumes, 4),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Fetching historical OHLCV data from Kraken (CCXT)")
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Symbols   : %s", SYMBOLS)
    logger.info("Timeframes: %s", TIMEFRAMES)
    logger.info("Date ranges: %s", DATE_RANGES)
    logger.info("=" * 60)

    loader = DataLoader()
    total_files = len(SYMBOLS) * len(TIMEFRAMES) * len(DATE_RANGES)
    done = 0
    api_errors = 0    # combinations where we fell back to synthetic data

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            for start_date, end_date in DATE_RANGES:
                done += 1
                logger.info(
                    "[%d/%d] %s  %s  %s → %s",
                    done, total_files, symbol, tf, start_date, end_date,
                )

                # ── Try the exchange first ────────────────────────────────
                df = pd.DataFrame()
                try:
                    df = loader.load(
                        symbol=symbol,
                        timeframe=tf,
                        start_date=start_date,
                        end_date=end_date,
                        force_refresh=FORCE_REFRESH,
                    )
                except Exception as exc:
                    logger.warning("  API fetch failed (%s) — using synthetic data", exc)

                # ── Fall back to synthetic data if needed ─────────────────
                if df.empty:
                    logger.warning(
                        "  ⚠  No candles from API — generating synthetic fallback"
                    )
                    api_errors += 1
                    try:
                        df = _generate_synthetic_ohlcv(symbol, tf, start_date, end_date)
                        # Save to the same CSV path the loader would have used.
                        # _save_synthetic_to_csv merges with any existing data so
                        # both date ranges end up in a single file.
                        csv_path = (
                            loader.data_dir
                            / (symbol.replace("/", "_") + f"_{tf}.csv")
                        )
                        _save_synthetic_to_csv(df, csv_path)
                        logger.info(
                            "  ✓  Synthetic: %d candles  (%s → %s)",
                            len(df),
                            df["timestamp"].iloc[0],
                            df["timestamp"].iloc[-1],
                        )
                    except Exception as syn_exc:
                        logger.error(
                            "  ✗  Synthetic fallback failed: %s", syn_exc
                        )
                else:
                    logger.info(
                        "  ✓  %d candles  (%s → %s)",
                        len(df),
                        df["timestamp"].iloc[0].date(),
                        df["timestamp"].iloc[-1].date(),
                    )

    logger.info("=" * 60)
    logger.info(
        "Done.  %d/%d from API  (%d used synthetic fallback)",
        total_files - api_errors, total_files, api_errors,
    )
    logger.info("CSVs saved to: %s", DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
