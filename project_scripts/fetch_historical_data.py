#!/usr/bin/env python3
"""
fetch_historical_data.py — Download OHLCV data from Kraken via CCXT.

Fetches candle data for all configured assets and timeframes over the
standard date ranges (2019–2021 and 2024–2025), then saves each dataset
as a CSV in data/historical/.

Usage
-----
    python project_scripts/fetch_historical_data.py

Optional arguments can be edited in the CONFIGURATION section below.
"""

import logging
import sys
from pathlib import Path

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
    errors = 0

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            for start_date, end_date in DATE_RANGES:
                done += 1
                logger.info(
                    "[%d/%d] %s  %s  %s → %s",
                    done, total_files, symbol, tf, start_date, end_date,
                )
                try:
                    df = loader.load(
                        symbol=symbol,
                        timeframe=tf,
                        start_date=start_date,
                        end_date=end_date,
                        force_refresh=FORCE_REFRESH,
                    )
                    if df.empty:
                        logger.warning("  ⚠  No candles returned")
                        errors += 1
                    else:
                        logger.info(
                            "  ✓  %d candles  (%s → %s)",
                            len(df),
                            df["timestamp"].iloc[0].date(),
                            df["timestamp"].iloc[-1].date(),
                        )
                except Exception as exc:
                    logger.error("  ✗  Failed: %s", exc)
                    errors += 1

    logger.info("=" * 60)
    logger.info(
        "Done.  %d/%d successful  (%d errors)",
        total_files - errors, total_files, errors,
    )
    logger.info("CSVs saved to: %s", DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
