"""
Visualisation for backtesting results.

Plots equity curves, drawdowns, and buy/sell entry/exit markers.
All plots are saved to the results directory; no interactive display is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import RESULTS_DIR

logger = logging.getLogger(__name__)

# Colour palette
_COLOURS = {
    "equity":   "#2196F3",   # blue
    "drawdown": "#F44336",   # red
    "buy":      "#4CAF50",   # green
    "sell":     "#FF5722",   # orange-red
    "price":    "#9E9E9E",   # grey
    "portfolio":"#673AB7",   # purple
}


def _save(fig: "plt.Figure", path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_equity_curve(
    equity_curve: Sequence[float],
    timestamps: Sequence | None = None,
    title: str = "Equity Curve",
    output_path: Path | None = None,
    initial_usd: float | None = None,
) -> Path:
    """
    Plot the portfolio equity curve over time.

    Parameters
    ----------
    equity_curve  : sequence of equity values (one per candle)
    timestamps    : optional sequence of datetime values for the x-axis
    title         : plot title
    output_path   : save location; defaults to RESULTS_DIR/equity_curve.png
    initial_usd   : draw a horizontal baseline at this value (optional)

    Returns
    -------
    Path to the saved image.
    """
    out = output_path or RESULTS_DIR / "equity_curve.png"

    fig, ax = plt.subplots(figsize=(14, 5))
    x = list(timestamps) if timestamps is not None else list(range(len(equity_curve)))
    ax.plot(x, equity_curve, color=_COLOURS["equity"], linewidth=1.5, label="Portfolio Equity")

    if initial_usd is not None:
        ax.axhline(initial_usd, color="black", linewidth=0.8, linestyle="--",
                   alpha=0.5, label=f"Initial ${initial_usd:,.0f}")

    if timestamps is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_xlabel("Date" if timestamps is not None else "Candle Index")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, out)
    return out


def plot_drawdowns(
    equity_curve: Sequence[float],
    timestamps: Sequence | None = None,
    title: str = "Drawdown Chart",
    output_path: Path | None = None,
) -> Path:
    """
    Plot the drawdown time-series.

    Returns
    -------
    Path to the saved image.
    """
    out = output_path or RESULTS_DIR / "drawdowns.png"
    curve = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max * 100  # in percent, negative values

    fig, ax = plt.subplots(figsize=(14, 4))
    x = list(timestamps) if timestamps is not None else list(range(len(dd)))
    ax.fill_between(x, dd, 0, color=_COLOURS["drawdown"], alpha=0.4)
    ax.plot(x, dd, color=_COLOURS["drawdown"], linewidth=1.0)

    if timestamps is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date" if timestamps is not None else "Candle Index")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, out)
    return out


def plot_price_with_signals(
    price_series: Sequence[float],
    trade_log: pd.DataFrame,
    timestamps: Sequence | None = None,
    asset: str = "BTC",
    title: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Plot price chart with buy/sell markers from the trade log.

    Parameters
    ----------
    price_series : close prices for the asset
    trade_log    : DataFrame from PortfolioSimulator.get_trade_log()
    timestamps   : datetime values for x-axis
    asset        : asset symbol (used to filter trade_log)
    title        : plot title
    output_path  : save location

    Returns
    -------
    Path to the saved image.
    """
    out = output_path or RESULTS_DIR / f"signals_{asset}.png"
    t = title or f"{asset} Price with Entry/Exit Signals"

    fig, ax = plt.subplots(figsize=(14, 5))
    x = list(timestamps) if timestamps is not None else list(range(len(price_series)))
    ax.plot(x, price_series, color=_COLOURS["price"], linewidth=1.0, label=f"{asset} Close")

    if not trade_log.empty and "asset" in trade_log.columns:
        asset_trades = trade_log[trade_log["asset"] == asset]

        # Map trade timestamps to x positions
        if timestamps is not None:
            ts_series = pd.Series(x)
            for _, row in asset_trades.iterrows():
                trade_ts = pd.Timestamp(row["timestamp"])
                # Find nearest index
                diffs = [(abs((pd.Timestamp(t) - trade_ts).total_seconds()), i)
                         for i, t in enumerate(timestamps)]
                if diffs:
                    _, idx = min(diffs)
                    yval = price_series[idx]
                    colour = _COLOURS["buy"] if row["side"] == "buy" else _COLOURS["sell"]
                    marker = "^" if row["side"] == "buy" else "v"
                    ax.scatter(x[idx], yval, color=colour, marker=marker, s=80, zorder=5)
        else:
            buys = asset_trades[asset_trades["side"] == "buy"]
            sells = asset_trades[asset_trades["side"] == "sell"]
            if not buys.empty:
                ax.scatter(buys.index.tolist(), [price_series[i] for i in buys.index],
                           color=_COLOURS["buy"], marker="^", s=80, label="Buy", zorder=5)
            if not sells.empty:
                ax.scatter(sells.index.tolist(), [price_series[i] for i in sells.index],
                           color=_COLOURS["sell"], marker="v", s=80, label="Sell", zorder=5)

    if timestamps is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    ax.set_title(t, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date" if timestamps is not None else "Candle Index")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save(fig, out)
    return out


def plot_full_dashboard(
    equity_curve: Sequence[float],
    trade_log: pd.DataFrame,
    price_series: Sequence[float] | None = None,
    timestamps: Sequence | None = None,
    asset: str = "BTC",
    title: str = "Backtest Dashboard",
    output_path: Path | None = None,
    initial_usd: float | None = None,
) -> Path:
    """
    Three-panel dashboard: equity curve, drawdowns, price + signals.

    Returns
    -------
    Path to the saved image.
    """
    out = output_path or RESULTS_DIR / "dashboard.png"
    n_panels = 3 if price_series is not None else 2

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
    x = list(timestamps) if timestamps is not None else list(range(len(equity_curve)))

    # Panel 1: equity curve
    ax0 = axes[0]
    ax0.plot(x, equity_curve, color=_COLOURS["equity"], linewidth=1.5)
    if initial_usd is not None:
        ax0.axhline(initial_usd, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax0.set_title("Portfolio Equity", fontsize=11)
    ax0.set_ylabel("USD")
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax0.grid(True, alpha=0.3)

    # Panel 2: drawdown
    curve = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max * 100
    ax1 = axes[1]
    ax1.fill_between(x, dd, 0, color=_COLOURS["drawdown"], alpha=0.4)
    ax1.plot(x, dd, color=_COLOURS["drawdown"], linewidth=1.0)
    ax1.set_title("Drawdown", fontsize=11)
    ax1.set_ylabel("%")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax1.grid(True, alpha=0.3)

    # Panel 3: price + signals (optional)
    if price_series is not None:
        ax2 = axes[2]
        ax2.plot(x, price_series, color=_COLOURS["price"], linewidth=1.0)
        ax2.set_title(f"{asset} Price + Signals", fontsize=11)
        ax2.set_ylabel("USD")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax2.grid(True, alpha=0.3)

        if not trade_log.empty and "asset" in trade_log.columns:
            at = trade_log[trade_log["asset"] == asset]
            buys = at[at["side"] == "buy"]
            sells = at[at["side"] == "sell"]
            if timestamps is not None:
                pass  # simplified — skip signal overlays in dashboard for now
            else:
                if not buys.empty:
                    bi = buys.index.tolist()
                    ax2.scatter(bi, [price_series[i] for i in bi if i < len(price_series)],
                                color=_COLOURS["buy"], marker="^", s=60, label="Buy", zorder=5)
                if not sells.empty:
                    si = sells.index.tolist()
                    ax2.scatter(si, [price_series[i] for i in si if i < len(price_series)],
                                color=_COLOURS["sell"], marker="v", s=60, label="Sell", zorder=5)
            ax2.legend(fontsize=8)

    if timestamps is not None:
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save(fig, out)
    return out
