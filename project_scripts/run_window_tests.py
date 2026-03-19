#!/usr/bin/env python3
"""
run_window_tests.py — Micro-level backtest runner.

Runs the bot on each bullish and bearish 4–8 hour test window and
produces per-window reports.  This is for debugging entry/exit logic,
trend detection, hedging, and RSI signals.

Output: results/full_backtests/window_*/

Usage
-----
    python project_scripts/run_window_tests.py

Prerequisites
-------------
    1. Run fetch_historical_data.py
    2. Run extract_test_windows.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.backtest.runner import BacktestRunner
from project.backtest.config import TEST_WINDOWS_DIR, INITIAL_USD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("MICRO-LEVEL WINDOW BACKTESTS")
    logger.info("Windows dir: %s", TEST_WINDOWS_DIR)
    logger.info("=" * 60)

    # Check that windows exist
    bull_dir = TEST_WINDOWS_DIR / "bull"
    bear_dir = TEST_WINDOWS_DIR / "bear"
    bull_csvs = sorted(bull_dir.glob("*.csv")) if bull_dir.exists() else []
    bear_csvs = sorted(bear_dir.glob("*.csv")) if bear_dir.exists() else []

    if not bull_csvs and not bear_csvs:
        logger.error(
            "No test windows found in %s. "
            "Run extract_test_windows.py first.",
            TEST_WINDOWS_DIR,
        )
        sys.exit(1)

    logger.info("Found %d bull windows, %d bear windows",
                len(bull_csvs), len(bear_csvs))

    runner = BacktestRunner(initial_usd=INITIAL_USD)
    all_results = runner.run_all_window_backtests(TEST_WINDOWS_DIR)

    # Detailed per-window summary
    print("\n" + "=" * 70)
    print("  MICRO-LEVEL WINDOW RESULTS")
    print("=" * 70)

    for regime in ("bull", "bear"):
        regime_results = all_results.get(regime, [])
        if not regime_results:
            print(f"\n  {regime.upper()}: No results")
            continue

        print(f"\n  {regime.upper()} WINDOWS ({len(regime_results)} tests)")
        print(f"  {'File':<40} {'Return':>8} {'Trades':>7} {'Win%':>6} {'MDD':>7}")
        print("  " + "-" * 70)

        for res in regime_results:
            m = res.get("metrics", {})
            fname = Path(res.get("window_file", "")).name[:38]
            print(
                f"  {fname:<40} "
                f"{m.get('total_return_pct', 0):>7.1f}% "
                f"{m.get('num_trades', 0):>7d} "
                f"{m.get('win_rate', 0):>5.0f}% "
                f"{m.get('max_drawdown_pct', 0):>6.1f}%"
            )

    print("\n" + "=" * 70)
    print("  Results saved to: results/full_backtests/window_*/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
