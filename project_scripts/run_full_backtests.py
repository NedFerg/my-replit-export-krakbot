#!/usr/bin/env python3
"""
run_full_backtests.py — Macro-level backtest runner.

Runs the bot through full historical cycles for 2019–2021 and 2023–2024
across the primary assets, producing:
    - Equity curves
    - Drawdown charts
    - Trade logs
    - Summary metrics

Output: results/full_backtests/{symbol}_{timeframe}_{years}/

Usage
-----
    python project_scripts/run_full_backtests.py

Prerequisites
-------------
    Run fetch_historical_data.py first to download data.

Configuration
-------------
    Edit SYMBOLS and TIMEFRAME below to customise what is tested.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.backtest.runner import BacktestRunner
from project.backtest.config import INITIAL_USD, DEFAULT_TIMEFRAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — edit to customise
# ---------------------------------------------------------------------------
SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
]
TIMEFRAME = DEFAULT_TIMEFRAME   # "1h"
DATE_RANGES = [
    ("2019-01-01", "2021-12-31"),
    ("2023-01-01", "2024-12-31"),
]


def main() -> None:
    logger.info("=" * 60)
    logger.info("MACRO-LEVEL FULL BACKTESTS")
    logger.info("Symbols    : %s", SYMBOLS)
    logger.info("Timeframe  : %s", TIMEFRAME)
    logger.info("Date ranges: %s", DATE_RANGES)
    logger.info("=" * 60)

    runner = BacktestRunner(initial_usd=INITIAL_USD, timeframe=TIMEFRAME)
    summary_rows = []

    total = len(SYMBOLS) * len(DATE_RANGES)
    done = 0
    errors = 0

    for symbol in SYMBOLS:
        for start_date, end_date in DATE_RANGES:
            done += 1
            label = f"{symbol.split('/')[0]}_{TIMEFRAME}_{start_date[:4]}-{end_date[:4]}"
            logger.info("")
            logger.info("[%d/%d] %s  %s → %s", done, total, symbol, start_date, end_date)
            logger.info("-" * 60)

            try:
                results = runner.run_full_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=TIMEFRAME,
                    save_results=True,
                )
                if results and "metrics" in results:
                    m = results["metrics"]
                    summary_rows.append({
                        "symbol": symbol,
                        "period": f"{start_date[:4]}–{end_date[:4]}",
                        "return_pct": m.get("total_return_pct", 0),
                        "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                        "sharpe": m.get("sharpe_ratio", 0),
                        "sortino": m.get("sortino_ratio", 0),
                        "trades": m.get("num_trades", 0),
                        "win_rate": m.get("win_rate", 0),
                    })
                else:
                    logger.warning("No results for %s %s–%s", symbol, start_date, end_date)
                    errors += 1
            except Exception as exc:
                logger.error("Error running %s %s → %s: %s", symbol, start_date, end_date, exc)
                errors += 1

    # Final summary table
    if summary_rows:
        print("\n" + "=" * 90)
        print("  MACRO BACKTEST SUMMARY")
        print("=" * 90)
        print(f"  {'Symbol':<12} {'Period':<14} {'Return':>9} {'MaxDD':>8} "
              f"{'Sharpe':>8} {'Sortino':>8} {'Trades':>7} {'Win%':>6}")
        print("  " + "-" * 88)
        for row in summary_rows:
            print(
                f"  {row['symbol']:<12} {row['period']:<14} "
                f"{row['return_pct']:>8.1f}% {row['max_drawdown_pct']:>7.1f}% "
                f"{row['sharpe']:>8.3f} {row['sortino']:>8.3f} "
                f"{row['trades']:>7d} {row['win_rate']:>5.0f}%"
            )
        print("=" * 90)
        print(f"\n  Completed: {done - errors}/{total}  ({errors} errors)")
        print("  Results saved to: results/full_backtests/")
        print("=" * 90 + "\n")
    else:
        logger.warning("No results generated. Check that data has been downloaded.")


if __name__ == "__main__":
    main()
