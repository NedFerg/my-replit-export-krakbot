#!/usr/bin/env python3
"""
run_window_tests.py — Micro-level backtest runner.

Runs the bot on each bullish and bearish 4–8 hour test window and
produces per-window reports.  This is for debugging entry/exit logic,
trend detection, hedging, and RSI signals.

Output: results/window_tests/

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
from project.backtest.config import TEST_WINDOWS_DIR, INITIAL_USD, WINDOW_TESTS_RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _write_trace(results: dict, out_dir: Path, label: str) -> None:
    """Write a detailed signal trace to a .txt file."""
    step_log = results.get("step_log")
    if step_log is None or step_log.empty:
        return
    trace_path = out_dir / f"{label}_trace.txt"
    lines = [f"Signal trace: {label}", "=" * 70, ""]
    for _, row in step_log.iterrows():
        action = ""
        if row.get("order_executed"):
            action = ">>> BUY" if row.get("signal", 0) == 1 else "<<< SELL"
        lines.append(
            f"{str(row['timestamp'])[:19]}  "
            f"close={row['close']:.2f}  "
            f"signal={int(row.get('signal', 0)):+d}  "
            f"rsi={str(row.get('rsi') or 'N/A'):>5}  "
            f"ema20={str(row.get('ema20') or 'N/A'):>8}  "
            f"equity={row.get('equity', 0):.2f}  "
            f"{action}"
        )
    trace_path.write_text("\n".join(lines) + "\n")
    logger.info("Trace written: %s", trace_path)


def main() -> None:
    logger.info("=" * 60)
    logger.info("MICRO-LEVEL WINDOW BACKTESTS")
    logger.info("Windows dir: %s", TEST_WINDOWS_DIR)
    logger.info("Results dir: %s", WINDOW_TESTS_RESULTS_DIR)
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

    runner = BacktestRunner(initial_usd=INITIAL_USD, output_dir=WINDOW_TESTS_RESULTS_DIR)
    all_results = runner.run_all_window_backtests(TEST_WINDOWS_DIR)

    # Write per-window trace files
    for regime in ("bull", "bear"):
        for i, res in enumerate(all_results.get(regime, []), 1):
            label = f"{regime}_{i:03d}"
            _write_trace(res, WINDOW_TESTS_RESULTS_DIR, label)

    # Build summary lines
    summary_lines: list[str] = []
    summary_lines.append("MICRO-LEVEL TEST RESULTS")
    summary_lines.append("=" * 80)
    summary_lines.append("")

    for regime in ("bull", "bear"):
        regime_results = all_results.get(regime, [])
        label_up = regime.upper()

        summary_lines.append(f"{label_up} WINDOWS:")
        header = (
            f"  {'Window':<8} {'File':<35} {'Return':>8} "
            f"{'Trades':>7} {'Win%':>6} {'MaxDD':>7}"
        )
        summary_lines.append(header)
        summary_lines.append("  " + "-" * 75)

        if not regime_results:
            summary_lines.append("  (no results)")
        else:
            returns = []
            win_rates = []
            for i, res in enumerate(regime_results, 1):
                m = res.get("metrics", {})
                fname = Path(res.get("window_file", "")).name[:33]
                ret = m.get("total_return_pct", 0.0)
                wr = m.get("win_rate", 0.0)
                returns.append(ret)
                win_rates.append(wr)
                summary_lines.append(
                    f"  {i:03d}     {fname:<35} "
                    f"{ret:>7.1f}% "
                    f"{m.get('num_trades', 0):>7d} "
                    f"{wr:>5.0f}% "
                    f"{m.get('max_drawdown_pct', 0):>6.1f}%"
                )
            avg_ret = sum(returns) / len(returns) if returns else 0.0
            avg_wr = sum(win_rates) / len(win_rates) if win_rates else 0.0
            summary_lines.append(
                f"\n  SUMMARY: {len(regime_results)} windows  |  "
                f"Avg return: {avg_ret:+.1f}%  |  Avg win rate: {avg_wr:.0f}%"
            )

        summary_lines.append("")

    summary_lines.append("=" * 80)
    summary_lines.append(f"Detailed traces: {WINDOW_TESTS_RESULTS_DIR}/{{bull|bear}}_NNN_trace.txt")
    summary_lines.append("=" * 80)

    # Print summary
    print("\n")
    for line in summary_lines:
        print(line)
    print()

    # Save MICRO_SUMMARY.txt
    summary_path = WINDOW_TESTS_RESULTS_DIR / "MICRO_SUMMARY.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
