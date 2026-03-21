"""
Top-level backtest runner — orchestrates data loading, engine execution,
metrics reporting, and plot generation for both window and full backtests.

Usage
-----
    from project.backtest.runner import BacktestRunner

    runner = BacktestRunner()
    runner.run_full_backtest("BTC/USD", "2023-01-01", "2023-12-31")
    runner.run_window_backtest("path/to/window.csv", "BTC")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest_engine import BacktestEngine
from .config import (
    BACKTEST_ASSETS,
    BACKTEST_TIMEFRAMES,
    DEFAULT_TIMEFRAME,
    INITIAL_USD,
    RESULTS_DIR,
    TAKER_FEE,
    SLIPPAGE,
)
from .data_loader import DataLoader
from .metrics import compute_all_metrics, print_metrics
from .plotter import plot_equity_curve, plot_drawdowns, plot_full_dashboard

logger = logging.getLogger(__name__)


def _symbol_to_short(symbol: str) -> str:
    """'BTC/USD' → 'BTC'"""
    return symbol.split("/")[0]


class BacktestRunner:
    """
    High-level orchestrator for running backtests.

    Parameters
    ----------
    initial_usd : starting portfolio value
    timeframe   : default OHLCV candle timeframe
    output_dir  : where results are saved (defaults to RESULTS_DIR)
    """

    def __init__(
        self,
        initial_usd: float = INITIAL_USD,
        timeframe: str = DEFAULT_TIMEFRAME,
        output_dir: Path | None = None,
    ) -> None:
        self.initial_usd = initial_usd
        self.timeframe = timeframe
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DataLoader()

    # ------------------------------------------------------------------
    # Full backtests (macro-level)
    # ------------------------------------------------------------------

    def run_full_backtest(
        self,
        symbol: str = "BTC/USD",
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        timeframe: str | None = None,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run a full backtest for *symbol* over the specified date range.

        Parameters
        ----------
        symbol      : trading pair, e.g. "BTC/USD"
        start_date  : ISO date string
        end_date    : ISO date string
        timeframe   : candle timeframe; defaults to self.timeframe
        save_results: write results to disk

        Returns
        -------
        dict with keys: metrics, trade_log, equity_curve_df, step_log
        """
        tf = timeframe or self.timeframe
        short = _symbol_to_short(symbol)

        logger.info("=" * 60)
        logger.info("Full backtest: %s  %s  %s → %s", symbol, tf, start_date, end_date)
        logger.info("=" * 60)

        df = self.loader.load(symbol, tf, start_date, end_date)
        if df.empty:
            logger.error("No data loaded for %s — aborting backtest", symbol)
            return {}

        engine = BacktestEngine(
            df_primary=df,
            primary_symbol=short,
            initial_usd=self.initial_usd,
            taker_fee=TAKER_FEE,
            slippage=SLIPPAGE,
            start_date=start_date,
            end_date=end_date,
        )
        results = engine.run()

        print_metrics(results["metrics"])

        if save_results:
            tag = f"{short}_{tf}_{start_date[:4]}-{end_date[:4]}"
            self._save_results(results, tag)

        return results

    def run_full_backtests_all_periods(
        self,
        symbol: str = "BTC/USD",
        timeframe: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run the full backtest for both standard date ranges (2019–2021 and 2023–2024).
        """
        tf = timeframe or self.timeframe
        periods = [
            ("2019-01-01", "2021-12-31"),
            ("2023-01-01", "2024-12-31"),
        ]
        all_results = []
        for start, end in periods:
            result = self.run_full_backtest(symbol, start, end, tf)
            all_results.append(result)
        return all_results

    # ------------------------------------------------------------------
    # Window backtests (micro-level)
    # ------------------------------------------------------------------

    def run_window_backtest(
        self,
        csv_path: str | Path,
        symbol: str = "BTC",
        label: str = "",
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run a backtest on a single test window CSV.

        Parameters
        ----------
        csv_path   : path to the window CSV file
        symbol     : asset symbol (short form, e.g. "BTC")
        label      : descriptive label used in output filenames
        save_results: write results to disk

        Returns
        -------
        dict with keys: metrics, trade_log, equity_curve_df, step_log
        """
        csv_path = Path(csv_path)
        logger.info("Window backtest: %s  (%s)", csv_path.name, label)

        df = self.loader._load_csv(csv_path)
        if df.empty:
            logger.error("No data in %s", csv_path)
            return {}

        engine = BacktestEngine(
            df_primary=df,
            primary_symbol=symbol,
            initial_usd=self.initial_usd,
            taker_fee=TAKER_FEE,
            slippage=SLIPPAGE,
        )
        results = engine.run()

        print_metrics(results["metrics"])

        if save_results:
            tag = f"window_{label or csv_path.stem}"
            self._save_results(results, tag)

        return results

    def run_all_window_backtests(
        self,
        windows_dir: str | Path | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Run backtests on all bull and bear windows.

        Parameters
        ----------
        windows_dir : parent directory containing bull/ and bear/ subdirs.
                      Defaults to data/test_windows/.

        Returns
        -------
        dict with keys "bull" and "bear", each containing a list of result dicts.
        """
        from .config import TEST_WINDOWS_DIR

        base = Path(windows_dir) if windows_dir else TEST_WINDOWS_DIR
        results: dict[str, list] = {"bull": [], "bear": []}

        for regime in ("bull", "bear"):
            regime_dir = base / regime
            csvs = sorted(regime_dir.glob("*.csv"))
            if not csvs:
                logger.warning("No CSVs found in %s", regime_dir)
                continue
            logger.info("%s windows: %d files", regime, len(csvs))
            for csv_path in csvs:
                try:
                    res = self.run_window_backtest(
                        csv_path,
                        symbol="BTC",
                        label=f"{regime}_{csv_path.stem}",
                    )
                    if res:
                        res["window_file"] = str(csv_path)
                        res["regime"] = regime
                        results[regime].append(res)
                except Exception as exc:
                    logger.error("Error running window %s: %s", csv_path, exc)

        # Print aggregate summary
        self._print_window_summary(results)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_results(self, results: dict[str, Any], tag: str) -> None:
        """Save metrics JSON, trade log CSV, equity curve CSV, and plots."""
        out = self.output_dir / tag
        out.mkdir(parents=True, exist_ok=True)

        # Metrics JSON (strip non-serialisable arrays)
        metrics_to_save = {
            k: v for k, v in results["metrics"].items()
            if k not in ("equity_curve", "drawdown_curve")
        }
        with open(out / "metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        # Trade log CSV
        if not results.get("trade_log", pd.DataFrame()).empty:
            results["trade_log"].to_csv(out / "trade_log.csv", index=False)

        # Equity curve CSV
        if not results.get("equity_curve_df", pd.DataFrame()).empty:
            results["equity_curve_df"].to_csv(out / "equity_curve.csv", index=False)

        # Step log CSV
        if not results.get("step_log", pd.DataFrame()).empty:
            results["step_log"].to_csv(out / "step_log.csv", index=False)

        # Plots
        eq_curve = results["metrics"].get("equity_curve", [])
        eq_df = results.get("equity_curve_df", pd.DataFrame())
        timestamps = eq_df["timestamp"].tolist() if not eq_df.empty else None
        tl = results.get("trade_log", pd.DataFrame())

        if eq_curve:
            plot_equity_curve(
                eq_curve,
                timestamps=timestamps,
                title=f"Equity Curve — {tag}",
                output_path=out / "equity_curve.png",
                initial_usd=self.initial_usd,
            )
            plot_drawdowns(
                eq_curve,
                timestamps=timestamps,
                title=f"Drawdown — {tag}",
                output_path=out / "drawdowns.png",
            )
            plot_full_dashboard(
                eq_curve,
                tl,
                timestamps=timestamps,
                title=f"Backtest Dashboard — {tag}",
                output_path=out / "dashboard.png",
                initial_usd=self.initial_usd,
            )

        logger.info("Results saved to: %s", out)

    @staticmethod
    def _print_window_summary(results: dict[str, list]) -> None:
        """Print an aggregate summary of window backtest results."""
        print("\n" + "=" * 60)
        print("  WINDOW BACKTEST AGGREGATE SUMMARY")
        print("=" * 60)
        for regime, res_list in results.items():
            if not res_list:
                print(f"  {regime.upper()}: no results")
                continue
            returns = [r["metrics"]["total_return_pct"] for r in res_list if r.get("metrics")]
            wins = [r for r in returns if r > 0]
            print(f"  {regime.upper()}: {len(res_list)} windows  |  "
                  f"avg return: {sum(returns)/len(returns):.1f}%  |  "
                  f"win rate: {len(wins)/len(returns)*100:.0f}%")
        print("=" * 60 + "\n")
