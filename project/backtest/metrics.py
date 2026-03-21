"""
Performance metrics for backtesting results.

Functions accept an equity curve (list/array of USD values sampled at each
candle) and a trade log (list of Trade objects or a DataFrame).

All return values are plain Python floats unless otherwise noted.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADING_HOURS_PER_YEAR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------


def total_return(equity_curve: Sequence[float]) -> float:
    """Total return as a decimal (e.g. 0.35 = 35 %)."""
    if len(equity_curve) < 2:
        return 0.0
    start, end = equity_curve[0], equity_curve[-1]
    if start == 0:
        return 0.0
    return (end - start) / start


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """
    Maximum peak-to-trough drawdown as a positive decimal.

    Returns 0.0 if the equity never fell below its running peak.
    """
    curve = np.array(equity_curve, dtype=float)
    if len(curve) < 2:
        return 0.0
    running_max = np.maximum.accumulate(curve)
    drawdown = (curve - running_max) / running_max
    return float(-drawdown.min())  # positive value


def drawdown_curve(equity_curve: Sequence[float]) -> np.ndarray:
    """Return the full drawdown time-series as a 1-D array of positive values."""
    curve = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max
    return -dd  # positive values


def sharpe_ratio(
    equity_curve: Sequence[float],
    candle_hours: float = 1.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    equity_curve  : portfolio value at each candle
    candle_hours  : length of each candle in hours (1 for 1h data)
    risk_free_rate: annualised risk-free rate (decimal)
    """
    curve = np.array(equity_curve, dtype=float)
    if len(curve) < 2:
        return 0.0
    returns = np.diff(curve) / curve[:-1]
    if returns.std() == 0:
        return 0.0
    candles_per_year = TRADING_HOURS_PER_YEAR / candle_hours
    rf_per_candle = (1 + risk_free_rate) ** (1 / candles_per_year) - 1
    excess = returns - rf_per_candle
    return float(excess.mean() / excess.std() * math.sqrt(candles_per_year))


def sortino_ratio(
    equity_curve: Sequence[float],
    candle_hours: float = 1.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """
    Annualised Sortino ratio (uses downside deviation only).
    """
    curve = np.array(equity_curve, dtype=float)
    if len(curve) < 2:
        return 0.0
    returns = np.diff(curve) / curve[:-1]
    candles_per_year = TRADING_HOURS_PER_YEAR / candle_hours
    rf_per_candle = (1 + risk_free_rate) ** (1 / candles_per_year) - 1
    excess = returns - rf_per_candle
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * math.sqrt(candles_per_year))


def calmar_ratio(
    equity_curve: Sequence[float],
    candle_hours: float = 1.0,
) -> float:
    """Annualised return divided by max drawdown."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    candles_per_year = TRADING_HOURS_PER_YEAR / candle_hours
    n_candles = len(equity_curve) - 1
    if n_candles <= 0:
        return 0.0
    ann_return = (1 + total_return(equity_curve)) ** (candles_per_year / n_candles) - 1
    return float(ann_return / mdd)


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------


def compute_trade_stats(trade_log: pd.DataFrame) -> dict:
    """
    Compute trade-level statistics from a trade log DataFrame.

    Expected columns: timestamp, asset, side, quantity, price, fee_usd,
                      usd_spent, slippage_pct
    """
    if trade_log.empty:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_duration_hours": 0.0,
            "total_fees_usd": 0.0,
        }

    total_fees = float(trade_log["fee_usd"].sum())
    buys = trade_log[trade_log["side"] == "buy"].copy()
    sells = trade_log[trade_log["side"] == "sell"].copy()

    # Pair buys and sells per asset to compute round-trip P&L
    round_trips: list[dict] = []
    for asset in trade_log["asset"].unique():
        asset_buys = buys[buys["asset"] == asset].reset_index(drop=True)
        asset_sells = sells[sells["asset"] == asset].reset_index(drop=True)
        n = min(len(asset_buys), len(asset_sells))
        for i in range(n):
            buy_row = asset_buys.iloc[i]
            sell_row = asset_sells.iloc[i]
            pnl = (sell_row["price"] - buy_row["price"]) * sell_row["quantity"]
            pnl -= buy_row["fee_usd"] + sell_row["fee_usd"]
            try:
                sell_ts = pd.Timestamp(sell_row["timestamp"])
                buy_ts = pd.Timestamp(buy_row["timestamp"])
                # Strip timezone info if mixed tz-naive/tz-aware
                if sell_ts.tzinfo is not None and buy_ts.tzinfo is None:
                    buy_ts = buy_ts.tz_localize(sell_ts.tzinfo)
                elif buy_ts.tzinfo is not None and sell_ts.tzinfo is None:
                    sell_ts = sell_ts.tz_localize(buy_ts.tzinfo)
                duration_hours = (sell_ts - buy_ts).total_seconds() / 3600
            except Exception:
                duration_hours = 0.0
            round_trips.append({"pnl": pnl, "duration_hours": duration_hours})

    if not round_trips:
        return {
            "num_trades": len(trade_log),
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_duration_hours": 0.0,
            "total_fees_usd": total_fees,
        }

    pnls = [t["pnl"] for t in round_trips]
    durations = [t["duration_hours"] for t in round_trips]
    wins = sum(1 for p in pnls if p > 0)

    return {
        "num_trades": len(round_trips),
        "win_rate": round(wins / len(pnls) * 100, 1),
        "avg_trade_pnl": round(float(np.mean(pnls)), 2),
        "avg_duration_hours": round(float(np.mean(durations)), 2),
        "total_fees_usd": round(total_fees, 2),
    }


# ---------------------------------------------------------------------------
# Full metrics summary
# ---------------------------------------------------------------------------


def compute_all_metrics(
    equity_curve: Sequence[float],
    trade_log: pd.DataFrame | None = None,
    candle_hours: float = 1.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Compute and return a comprehensive dictionary of performance metrics.

    Parameters
    ----------
    equity_curve : portfolio value at each candle step
    trade_log    : DataFrame from PortfolioSimulator.get_trade_log()
    candle_hours : candle duration in hours (1 for 1h data, 4 for 4h, etc.)
    risk_free_rate: annualised risk-free rate for Sharpe / Sortino
    """
    curve = list(equity_curve)
    tl = trade_log if trade_log is not None else pd.DataFrame()

    metrics: dict = {}

    # Equity-curve metrics
    metrics["total_return_pct"] = round(total_return(curve) * 100, 2)
    metrics["max_drawdown_pct"] = round(max_drawdown(curve) * 100, 2)
    metrics["sharpe_ratio"] = round(sharpe_ratio(curve, candle_hours, risk_free_rate), 3)
    metrics["sortino_ratio"] = round(sortino_ratio(curve, candle_hours, risk_free_rate), 3)
    metrics["calmar_ratio"] = round(calmar_ratio(curve, candle_hours), 3)
    metrics["equity_start"] = round(curve[0], 2) if curve else 0.0
    metrics["equity_end"] = round(curve[-1], 2) if curve else 0.0
    metrics["equity_curve"] = curve  # raw array for plotting
    metrics["drawdown_curve"] = drawdown_curve(curve).tolist()

    # Trade metrics
    trade_stats = compute_trade_stats(tl)
    metrics.update(trade_stats)

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dict to stdout."""
    print("\n" + "=" * 52)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 52)
    print(f"  Total Return     : {metrics.get('total_return_pct', 0):>8.2f} %")
    print(f"  Max Drawdown     : {metrics.get('max_drawdown_pct', 0):>8.2f} %")
    print(f"  Sharpe Ratio     : {metrics.get('sharpe_ratio', 0):>8.3f}")
    print(f"  Sortino Ratio    : {metrics.get('sortino_ratio', 0):>8.3f}")
    print(f"  Calmar Ratio     : {metrics.get('calmar_ratio', 0):>8.3f}")
    print(f"  Total Trades     : {metrics.get('num_trades', 0):>8d}")
    print(f"  Win Rate         : {metrics.get('win_rate', 0):>8.1f} %")
    print(f"  Avg Trade P&L    : ${metrics.get('avg_trade_pnl', 0):>8.2f}")
    print(f"  Avg Duration     : {metrics.get('avg_duration_hours', 0):>8.2f} h")
    print(f"  Total Fees       : ${metrics.get('total_fees_usd', 0):>8.2f}")
    print(f"  Starting Equity  : ${metrics.get('equity_start', 0):>10,.2f}")
    print(f"  Ending Equity    : ${metrics.get('equity_end', 0):>10,.2f}")
    print("=" * 52 + "\n")
