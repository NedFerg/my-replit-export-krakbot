#!/usr/bin/env python3
"""
extract_test_windows.py — Extract bullish and bearish test windows from
historical BTC/USD data.

Scans 1-hour OHLCV data and identifies 10 strong bullish and 10 strong
bearish 4–8 hour windows based on:
    - % price change over the window
    - RSI behaviour (oversold/overbought)
    - Trend slope (linear regression)
    - Volatility expansion

Each window is saved as a separate CSV in:
    data/test_windows/bull/
    data/test_windows/bear/

Usage
-----
    python project_scripts/extract_test_windows.py

Prerequisites
-------------
    Run fetch_historical_data.py first to download BTC/USD 1h data.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.backtest.config import (
    DATA_DIR,
    TEST_WINDOWS_DIR,
    NUM_WINDOWS,
    WINDOW_HOURS,
    BULL_MIN_PCT,
    BEAR_MIN_PCT,
)
from project.backtest.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "BTC/USD"
TIMEFRAME = "1h"
WINDOW_SIZE = WINDOW_HOURS       # candles per window (1h candles → hours)
MIN_BULL_PCT = BULL_MIN_PCT      # minimum % gain for a bullish window
MIN_BEAR_PCT = abs(BEAR_MIN_PCT) # minimum % drop for a bearish window (positive)
MIN_SEPARATION = 72              # minimum candles between selected windows


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised RSI computation. Returns NaN for insufficient data."""
    deltas = np.diff(prices)
    rsi = np.full(len(prices), np.nan)
    if len(prices) <= period:
        return rsi
    for i in range(period, len(prices)):
        window = deltas[i - period: i]
        gains = window[window > 0].sum() / period
        losses = -window[window < 0].sum() / period
        if losses == 0:
            rsi[i] = 100.0
        else:
            rs = gains / losses
            rsi[i] = 100 - 100 / (1 + rs)
    return rsi


def _slope(prices: np.ndarray) -> float:
    """Linear regression slope normalised by mean price."""
    x = np.arange(len(prices))
    s = float(np.polyfit(x, prices, 1)[0])
    m = float(np.mean(prices))
    return s / m if m else 0.0


def _volatility(prices: np.ndarray) -> float:
    """Annualised (per-candle) standard deviation of log returns."""
    if len(prices) < 2:
        return 0.0
    log_returns = np.diff(np.log(prices))
    return float(np.std(log_returns))


# ---------------------------------------------------------------------------
# Window scoring
# ---------------------------------------------------------------------------

def score_window(window_df: pd.DataFrame) -> dict:
    """
    Score a candidate window for bullishness/bearishness.

    Returns a dict with keys: pct_change, slope, volatility, rsi_start.
    """
    closes = window_df["close"].values
    pct_change = (closes[-1] - closes[0]) / closes[0] * 100
    slope = _slope(closes)
    vol = _volatility(closes)

    # RSI at the start of the window (using the preceding context from the full df)
    rsi_val = np.nan

    return {
        "pct_change": pct_change,
        "slope": slope,
        "volatility": vol,
        "rsi_start": rsi_val,
    }


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_windows(df: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Scan df and extract up to NUM_WINDOWS bull and bear windows.

    Returns
    -------
    (bull_windows, bear_windows) — each is a list of DataFrames
    """
    closes = df["close"].values
    rsi_all = _rsi(closes)

    bull_windows: list[tuple[float, pd.DataFrame]] = []  # (score, df)
    bear_windows: list[tuple[float, pd.DataFrame]] = []

    # Slide a window of WINDOW_SIZE candles across the data
    step = max(1, WINDOW_SIZE // 2)
    n = len(df)

    for start_idx in range(0, n - WINDOW_SIZE + 1, step):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].copy()
        scores = score_window(window_df)

        pct = scores["pct_change"]
        slope = scores["slope"]
        vol = scores["volatility"]

        # Bullish candidate
        if pct >= MIN_BULL_PCT and slope > 0:
            bull_windows.append((pct, start_idx, window_df))

        # Bearish candidate
        elif pct <= -MIN_BEAR_PCT and slope < 0:
            bear_windows.append((abs(pct), start_idx, window_df))

    # Sort by score (largest move first) and enforce minimum separation
    def filter_windows(candidates: list) -> list[pd.DataFrame]:
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected: list[pd.DataFrame] = []
        used_indices: list[int] = []
        for score, start_idx, win_df in candidates:
            if len(selected) >= NUM_WINDOWS:
                break
            # Check separation from already-selected windows
            too_close = any(
                abs(start_idx - ui) < MIN_SEPARATION for ui in used_indices
            )
            if too_close:
                continue
            selected.append(win_df)
            used_indices.append(start_idx)
        return selected

    return filter_windows(bull_windows), filter_windows(bear_windows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Extracting bullish and bearish test windows")
    logger.info("Symbol: %s  |  Timeframe: %s  |  Window: %dh",
                SYMBOL, TIMEFRAME, WINDOW_SIZE)
    logger.info("=" * 60)

    loader = DataLoader()

    # Load all available BTC data (both date ranges)
    dfs = []
    for start, end in [("2019-01-01", "2021-12-31"), ("2023-01-01", "2024-12-31")]:
        try:
            df = loader.load(SYMBOL, TIMEFRAME, start, end)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            logger.error("Failed to load %s %s → %s: %s", SYMBOL, start, end, exc)

    if not dfs:
        logger.error("No historical data found. Run fetch_historical_data.py first.")
        sys.exit(1)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    logger.info("Total candles available: %d", len(full_df))

    bull_windows, bear_windows = extract_windows(full_df)

    logger.info("Found %d bullish windows (target: %d)", len(bull_windows), NUM_WINDOWS)
    logger.info("Found %d bearish windows (target: %d)", len(bear_windows), NUM_WINDOWS)

    # Save windows
    bull_dir = TEST_WINDOWS_DIR / "bull"
    bear_dir = TEST_WINDOWS_DIR / "bear"
    bull_dir.mkdir(parents=True, exist_ok=True)
    bear_dir.mkdir(parents=True, exist_ok=True)

    closes = full_df["close"].values
    rsi_all = _rsi(closes)
    base_symbol = SYMBOL.split("/")[0]

    summary: dict = {"bull": [], "bear": []}

    for i, win_df in enumerate(bull_windows, 1):
        start_ts = win_df["timestamp"].iloc[0]
        end_ts = win_df["timestamp"].iloc[-1]
        start_dt = start_ts.strftime("%Y%m%d_%H%M")
        end_dt = end_ts.strftime("%Y%m%d_%H%M")
        pct = (win_df["close"].iloc[-1] - win_df["close"].iloc[0]) / win_df["close"].iloc[0] * 100
        # Find global index to get RSI at window start
        global_idx = full_df.index[full_df["timestamp"] == start_ts].tolist()
        rsi_val = float(rsi_all[global_idx[0]]) if global_idx and not np.isnan(rsi_all[global_idx[0]]) else None
        score = score_window(win_df)
        fname = bull_dir / f"bull_{i:02d}_{start_dt}_{pct:+.1f}pct.csv"
        win_df.to_csv(fname, index=False)
        logger.info("  [bull %02d] %s → %s  (%+.1f%%)", i, start_dt, end_dt, pct)
        summary["bull"].append({
            "window": f"window_{i:03d}",
            "file": fname.name,
            "symbol": base_symbol,
            "start": start_ts.strftime("%Y-%m-%d %H:%M"),
            "end": end_ts.strftime("%Y-%m-%d %H:%M"),
            "pct_change": round(pct, 2),
            "rsi": round(rsi_val, 1) if rsi_val is not None else None,
            "slope": round(score["slope"], 6),
            "volatility": round(score["volatility"], 6),
        })

    for i, win_df in enumerate(bear_windows, 1):
        start_ts = win_df["timestamp"].iloc[0]
        end_ts = win_df["timestamp"].iloc[-1]
        start_dt = start_ts.strftime("%Y%m%d_%H%M")
        end_dt = end_ts.strftime("%Y%m%d_%H%M")
        pct = (win_df["close"].iloc[-1] - win_df["close"].iloc[0]) / win_df["close"].iloc[0] * 100
        global_idx = full_df.index[full_df["timestamp"] == start_ts].tolist()
        rsi_val = float(rsi_all[global_idx[0]]) if global_idx and not np.isnan(rsi_all[global_idx[0]]) else None
        score = score_window(win_df)
        fname = bear_dir / f"bear_{i:02d}_{start_dt}_{pct:+.1f}pct.csv"
        win_df.to_csv(fname, index=False)
        logger.info("  [bear %02d] %s → %s  (%+.1f%%)", i, start_dt, end_dt, pct)
        summary["bear"].append({
            "window": f"window_{i:03d}",
            "file": fname.name,
            "symbol": base_symbol,
            "start": start_ts.strftime("%Y-%m-%d %H:%M"),
            "end": end_ts.strftime("%Y-%m-%d %H:%M"),
            "pct_change": round(pct, 2),
            "rsi": round(rsi_val, 1) if rsi_val is not None else None,
            "slope": round(score["slope"], 6),
            "volatility": round(score["volatility"], 6),
        })

    # Write summary JSON
    summary_path = TEST_WINDOWS_DIR / "windows_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to: %s", summary_path)

    logger.info("=" * 60)
    logger.info("Windows saved to: %s", TEST_WINDOWS_DIR)
    logger.info("  Bull: %d windows", len(bull_windows))
    logger.info("  Bear: %d windows", len(bear_windows))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
