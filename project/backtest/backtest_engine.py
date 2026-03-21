"""
BacktestEngine — the core backtesting loop.

Feeds historical OHLCV candles through a simplified signal-generation
pipeline that mirrors the live bot's indicator logic, then executes
simulated trades via PortfolioSimulator.

The engine is intentionally standalone so it can be validated before
wiring in the full live-bot agent.  Indicator logic is intentionally
kept simple and extensible.

Usage
-----
    from project.backtest.backtest_engine import BacktestEngine

    engine = BacktestEngine(
        df_primary=df_btc,
        other_assets={"ETH": df_eth},
        initial_usd=10_000,
    )
    results = engine.run()
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    INITIAL_USD,
    TAKER_FEE,
    SLIPPAGE,
    MAX_ETF_EXPOSURE,
    RESULTS_DIR,
)
from .historical_feed import HistoricalDataFeed
from .portfolio_simulator import PortfolioSimulator
from .metrics import compute_all_metrics, print_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _rsi(prices: list[float], period: int = 14) -> float | None:
    """Return the current RSI value, or None if insufficient data."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices[-period - 1:])
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - 100 / (1 + rs))


def _bollinger_bands(
    prices: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[float, float, float] | None:
    """Return (upper, middle, lower) Bollinger Bands, or None."""
    if len(prices) < period:
        return None
    window = prices[-period:]
    mean = np.mean(window)
    std = np.std(window, ddof=1)
    return float(mean + std_dev * std), float(mean), float(mean - std_dev * std)


def _ema(prices: list[float], period: int) -> float | None:
    """Return the current EMA value, or None if insufficient data."""
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return float(ema)


def _trend_slope(prices: list[float], lookback: int = 20) -> float:
    """
    Return the linear regression slope over the last *lookback* prices,
    normalised by the mean price (% per candle).
    """
    if len(prices) < lookback:
        return 0.0
    window = prices[-lookback:]
    x = np.arange(lookback)
    slope = float(np.polyfit(x, window, 1)[0])
    mean_price = float(np.mean(window))
    return slope / mean_price if mean_price else 0.0


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


class SignalGenerator:
    """
    Lightweight signal generator that mirrors the live bot's indicator logic.

    Signals
    -------
    +1 : bullish — buy / increase position
     0 : neutral — hold / no action
    -1 : bearish — sell / reduce position or hedge
    """

    # Tunable thresholds
    RSI_OVERSOLD   = 30
    RSI_OVERBOUGHT = 70
    TREND_STRONG   = 0.0010   # 0.10% per candle
    TREND_WEAK     = -0.0005  # slope below this = downtrend

    def __init__(self) -> None:
        self._closes: list[float] = []

    def update(self, candle: dict) -> None:
        self._closes.append(candle["close"])

    def get_signal(self) -> tuple[int, dict]:
        """
        Return (signal, indicators) where signal ∈ {-1, 0, 1}.

        indicators : dict of current indicator values for logging.
        """
        closes = self._closes
        indicators: dict[str, Any] = {}

        rsi = _rsi(closes)
        bb = _bollinger_bands(closes)
        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        slope = _trend_slope(closes, 20)

        indicators["rsi"] = round(rsi, 1) if rsi is not None else None
        indicators["bb_upper"] = round(bb[0], 2) if bb else None
        indicators["bb_mid"] = round(bb[1], 2) if bb else None
        indicators["bb_lower"] = round(bb[2], 2) if bb else None
        indicators["ema20"] = round(ema20, 2) if ema20 is not None else None
        indicators["ema50"] = round(ema50, 2) if ema50 is not None else None
        indicators["slope"] = round(slope * 100, 4) if slope is not None else None

        if len(closes) < 50:
            return 0, indicators  # not enough data yet

        current = closes[-1]

        # Strong uptrend confirmation
        bullish_trend = (
            ema20 is not None
            and ema50 is not None
            and ema20 > ema50
            and slope > self.TREND_STRONG
        )

        # Strong downtrend
        bearish_trend = (
            ema20 is not None
            and ema50 is not None
            and ema20 < ema50
            and slope < self.TREND_WEAK
        )

        # Oversold bounce entry
        oversold = rsi is not None and rsi < self.RSI_OVERSOLD
        if bb:
            oversold = oversold or current < bb[2]

        # Overbought exit signal
        overbought = rsi is not None and rsi > self.RSI_OVERBOUGHT
        if bb:
            overbought = overbought or current > bb[0]

        # Final signal logic
        if oversold and not bearish_trend:
            return 1, indicators   # oversold bounce — buy
        if overbought and bearish_trend:
            return -1, indicators  # overbought in downtrend — sell/hedge
        if bullish_trend:
            return 1, indicators   # uptrend continuation — hold / buy
        if bearish_trend:
            return -1, indicators  # downtrend — sell / hedge
        return 0, indicators       # neutral — hold


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Core backtesting loop.

    Parameters
    ----------
    df_primary : DataFrame of OHLCV data for the primary trading asset
    primary_symbol : symbol name (e.g. "BTC")
    other_assets : dict mapping symbol → DataFrame for additional assets
    initial_usd : starting cash
    taker_fee : fractional taker fee
    slippage : fractional slippage
    start_date / end_date : date range slice for the feed
    position_size : fraction of equity to deploy per trade (0.0 – 1.0)
    output_dir : where to save results
    """

    def __init__(
        self,
        df_primary: pd.DataFrame,
        primary_symbol: str = "BTC",
        other_assets: dict[str, pd.DataFrame] | None = None,
        initial_usd: float = INITIAL_USD,
        taker_fee: float = TAKER_FEE,
        slippage: float = SLIPPAGE,
        start_date: str | None = None,
        end_date: str | None = None,
        position_size: float = 0.5,
        output_dir: Path | None = None,
    ) -> None:
        self.primary_symbol = primary_symbol
        self.other_assets = other_assets or {}
        self.position_size = min(max(position_size, 0.0), 1.0)
        self.output_dir = output_dir or RESULTS_DIR

        # Historical feed for primary asset
        self.feed = HistoricalDataFeed(df_primary, start_date, end_date)

        # Secondary feeds (aligned by timestamp)
        self.secondary_feeds: dict[str, pd.DataFrame] = {}
        for sym, df in self.other_assets.items():
            feed = HistoricalDataFeed(df, start_date, end_date)
            self.secondary_feeds[sym] = feed.as_dataframe().set_index("timestamp")

        # Portfolio simulator
        all_symbols = [primary_symbol] + list(self.other_assets.keys())
        self.portfolio = PortfolioSimulator(
            initial_usd=initial_usd,
            assets=all_symbols,
            taker_fee=taker_fee,
            slippage=slippage,
        )

        # Signal generators per asset
        self.signal_gen = SignalGenerator()
        self.secondary_signal_gens: dict[str, SignalGenerator] = {
            sym: SignalGenerator() for sym in self.other_assets
        }

        # State tracking
        self._in_position = False
        self._candle_index = 0
        self._log_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full backtest.

        Returns
        -------
        dict with keys:
            metrics   — dict from compute_all_metrics()
            trade_log — pd.DataFrame
            equity_curve_df — pd.DataFrame with columns [timestamp, equity]
        """
        logger.info(
            "BacktestEngine: starting %s backtest  (%d candles)",
            self.primary_symbol, self.feed.total(),
        )
        self.feed.reset()

        prev_signal = 0

        for candle in self.feed:
            ts = candle["timestamp"]
            close = candle["close"]
            idx = candle["index"]

            # Update signal generator with this candle
            self.signal_gen.update(candle)
            signal, indicators = self.signal_gen.get_signal()

            # Build current prices dict
            prices = {self.primary_symbol: close}
            for sym, df in self.secondary_feeds.items():
                if ts in df.index:
                    prices[sym] = float(df.loc[ts, "close"])
                elif not df.empty:
                    # Use last known price
                    past = df[df.index <= ts]
                    if not past.empty:
                        prices[sym] = float(past.iloc[-1]["close"])

            self.portfolio.set_prices(prices)

            # Execute order on signal change (avoid churning on flat signals)
            order_executed = False
            if signal != prev_signal:
                trade = self._execute_signal(signal, ts, close)
                order_executed = trade is not None

            prev_signal = signal

            # Record equity
            self.portfolio.record_equity(ts, prices)

            # Log this step
            self._log_rows.append({
                "timestamp": ts,
                "close": close,
                "signal": signal,
                "rsi": indicators.get("rsi"),
                "bb_upper": indicators.get("bb_upper"),
                "bb_lower": indicators.get("bb_lower"),
                "ema20": indicators.get("ema20"),
                "ema50": indicators.get("ema50"),
                "slope": indicators.get("slope"),
                "in_position": self._in_position,
                "equity": self.portfolio.get_total_equity(prices),
                "order_executed": order_executed,
            })

            self._candle_index += 1

        # Finalise: close any open position
        if self._in_position:
            self.portfolio.close_position(self.primary_symbol)

        # Compute results
        equity_df = self.portfolio.get_equity_curve_df()
        trade_log = self.portfolio.get_trade_log()
        equity_values = equity_df["equity"].tolist() if not equity_df.empty else [INITIAL_USD]

        metrics = compute_all_metrics(
            equity_values,
            trade_log,
            candle_hours=self._detect_candle_hours(),
        )

        logger.info("BacktestEngine: finished.  Return=%.1f%%  Trades=%d",
                    metrics["total_return_pct"], metrics["num_trades"])

        return {
            "metrics": metrics,
            "trade_log": trade_log,
            "equity_curve_df": equity_df,
            "step_log": pd.DataFrame(self._log_rows),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_signal(
        self,
        signal: int,
        timestamp: datetime,
        close: float,
    ):
        """Convert a signal into a simulated trade."""
        equity = self.portfolio.get_total_equity()

        if signal == 1 and not self._in_position:
            # Enter long position
            usd_to_spend = equity * self.position_size
            if usd_to_spend > self.portfolio.usd_balance:
                usd_to_spend = self.portfolio.usd_balance * 0.99
            if usd_to_spend < 10:
                return None
            trade = self.portfolio.buy(
                self.primary_symbol,
                usd_amount=usd_to_spend,
                timestamp=timestamp,
            )
            if trade:
                self._in_position = True
                logger.debug("ENTER LONG  %s  @ %.2f", self.primary_symbol, close)
            return trade

        if signal == -1 and self._in_position:
            # Exit long position
            trade = self.portfolio.close_position(self.primary_symbol, timestamp=timestamp)
            if trade:
                self._in_position = False
                logger.debug("EXIT LONG   %s  @ %.2f", self.primary_symbol, close)
            return trade

        return None

    def _detect_candle_hours(self) -> float:
        """Infer candle duration in hours from the feed timestamps."""
        df = self.feed.as_dataframe()
        if len(df) < 2:
            return 1.0
        delta = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()
        return delta / 3600
