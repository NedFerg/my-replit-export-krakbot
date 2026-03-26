#!/usr/bin/env python3
"""
Enhanced Trading Bot v2 - Mean Reversion Swing Trader
Uses RSI + MACD + Support/Resistance for entries/exits
Tracks existing positions and pyramids into them

Supports full bearish signal logic (SELL / short ETF rotation) and
enforces market-hours guard for SETH/ETHD ETF trading.
"""

import logging
import os
from datetime import datetime
import pandas as pd

from project.config.ma_strategy_config import STRATEGY_PARAMS, INITIAL_CAPITAL_USD
from project.strategies.volume_climax_detector import VolumeClimaxDetector
from project.utils.market_hours import MarketHours, MarketSession
from project_scripts.technical_indicators import TechnicalIndicators
from project_scripts.position_manager import PositionManager, Position

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature toggle: short ETF trading (SETH / ETHD)
# Default ON — disable with: export ENABLE_SHORT_ETF_TRADING=false
# ---------------------------------------------------------------------------
_ENABLE_SHORT_ETF = os.getenv("ENABLE_SHORT_ETF_TRADING", "true").lower() not in (
    "false", "0", "no"
)

_market_hours = MarketHours()


def _etf_market_open() -> bool:
    """Return True only during regular US market hours (M-F 9:30–16:00 ET)."""
    return _market_hours.get_session() == MarketSession.REGULAR


class EnhancedTradeBot:
    """Mean reversion swing trading bot with full bull/bear signal logic."""

    def __init__(
        self,
        asset_name,
        kraken_pair,
        existing_amount=0,
        current_price=0,
        paper_trading=True,
    ):
        self.asset = asset_name
        self.pair = kraken_pair
        self.paper_trading = paper_trading
        self.params = STRATEGY_PARAMS.get(kraken_pair, {})

        # Position management
        self.position_manager = PositionManager(
            asset_name=asset_name,
            kraken_code=kraken_pair,
            existing_amount=existing_amount,
            current_price=current_price,
        )

        # Trading parameters (overridden by runner via capital_per_trade property)
        self.capital_per_trade = INITIAL_CAPITAL_USD
        self.max_position_size = max(existing_amount * 2, 1.0)
        self.stop_loss_pct = 5.0
        self.take_profit_pct = 15.0

        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_signal_threshold = 0.0001
        # Volume capitulation detector
        self.volume_detector = VolumeClimaxDetector(
            volume_spike_multiplier=2.5,
            wick_ratio_threshold=0.03,
            lookback_periods=50,
        )

        # State tracking
        self.last_signal = "WAIT"
        self.last_rsi = 50.0
        self.last_macd: dict = {}
        self.entry_history: list = []

        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(
            "[%s] %s bot initialised | existing=%.8f | etf_shorts=%s",
            mode,
            asset_name,
            existing_amount,
            "ON" if _ENABLE_SHORT_ETF else "OFF",
        )

    # ------------------------------------------------------------------
    # Confidence scoring weights and thresholds (class-level constants)
    # ------------------------------------------------------------------
    # Each factor contributes a fraction of the total score (must sum to 1.0)
    _CONF_WEIGHT_RSI        = 0.35  # depth into oversold/overbought zone
    _CONF_WEIGHT_MACD       = 0.25  # MACD histogram magnitude
    _CONF_WEIGHT_MA         = 0.20  # MA alignment (trend direction)
    _CONF_WEIGHT_SR         = 0.20  # proximity to support/resistance level

    _CONF_MACD_SCALE        = 0.01  # MACD histogram value that equals full score
    _CONF_SR_MAX_DEVIATION  = 0.05  # max price deviation (5%) from S/R for full score

    # ------------------------------------------------------------------
    # Signal calculation
    # ------------------------------------------------------------------

    def _bull_confidence(self, rsi, macd, fast_ma, slow_ma, sr, current_price) -> float:
        """
        Score bullish signal quality 0.0–1.0.

        Factors: RSI depth into oversold zone, MACD histogram magnitude,
        MA alignment, proximity to support.
        """
        score = 0.0
        # RSI: deeper oversold = higher confidence
        if rsi < self.rsi_oversold:
            score += min((self.rsi_oversold - rsi) / self.rsi_oversold, 1.0) * self._CONF_WEIGHT_RSI
        # MACD histogram positive and growing
        hist = macd.get("histogram", 0)
        if hist > 0:
            score += min(hist / self._CONF_MACD_SCALE, 1.0) * self._CONF_WEIGHT_MACD
        # MA uptrend
        if fast_ma > slow_ma:
            score += self._CONF_WEIGHT_MA
        # Price near support
        support = sr.get("support", 0)
        if support > 0 and current_price > 0:
            deviation = min(abs(current_price - support) / current_price, self._CONF_SR_MAX_DEVIATION)
            proximity = 1 - deviation / self._CONF_SR_MAX_DEVIATION
            score += proximity * self._CONF_WEIGHT_SR
        return round(min(score, 1.0), 3)

    def _bear_confidence(self, rsi, macd, fast_ma, slow_ma, sr, current_price) -> float:
        """
        Score bearish signal quality 0.0–1.0.

        Mirrors _bull_confidence logic for the short side.
        """
        score = 0.0
        # RSI: deeper overbought = higher confidence
        if rsi > self.rsi_overbought:
            score += min((rsi - self.rsi_overbought) / (100 - self.rsi_overbought), 1.0) * self._CONF_WEIGHT_RSI
        # MACD histogram negative and falling
        hist = macd.get("histogram", 0)
        if hist < 0:
            score += min(abs(hist) / self._CONF_MACD_SCALE, 1.0) * self._CONF_WEIGHT_MACD
        # MA downtrend
        if fast_ma < slow_ma:
            score += self._CONF_WEIGHT_MA
        # Price near resistance
        resistance = sr.get("resistance", 0)
        if resistance > 0 and current_price > 0:
            deviation = min(abs(current_price - resistance) / current_price, self._CONF_SR_MAX_DEVIATION)
            proximity = 1 - deviation / self._CONF_SR_MAX_DEVIATION
            score += proximity * self._CONF_WEIGHT_SR
        return round(min(score, 1.0), 3)

    def calculate_signals(self, df) -> dict:
        """Calculate all trading signals with volume capitulation priority."""
        if len(df) < 50:
            return {
                "signal": "WAIT",
                "reason": "Insufficient data",
                "rsi": None,
                "macd": {},
                "price": None,
                "confidence": 0.0,
                "etf_short_eligible": False,
            }

        # Technical indicators
        rsi = TechnicalIndicators.calculate_rsi(df, period=14)
        macd = TechnicalIndicators.calculate_macd(df)
        sr = TechnicalIndicators.find_support_resistance(df, lookback=20)
        atr = TechnicalIndicators.calculate_atr(df, period=14)

        fast_ma = df["close"].tail(self.params.get("fast_ma", 5)).mean()
        slow_ma = df["close"].tail(self.params.get("slow_ma", 50)).mean()
        current_price = float(df["close"].iloc[-1])

        self.last_rsi = rsi
        self.last_macd = macd

        # Determine whether ETF short is eligible right now
        etf_short_eligible = _ENABLE_SHORT_ETF and _etf_market_open()

        # ========== VOLUME CAPITULATION PRIORITY ==========
        # Check for capitulation BUY first (institutional buyers stepping in)
        cap_buy, cap_conf, cap_details = self.volume_detector.detect_capitulation_buy(df, rsi, current_price)
        if cap_buy and cap_conf >= 0.60:
            self.last_signal = "BUY"
            return {
                "signal": "BUY",
                "reason": (
                    f"[VOLUME CLIMAX] {self.asset} CAPITULATION BUY\n"
                    f"  RSI={rsi:.1f} (deep oversold)\n"
                    f"  Volume={cap_details.get('volume_ratio', 0):.1f}x avg (capitulation spike)\n"
                    f"  Lower Wick={cap_details.get('lower_wick_pct', 0):.1f}% (rejection of lower prices)\n"
                    f"  Confidence={cap_conf:.2f} ✅"
                ),
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price,
                "confidence": cap_conf,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "trend": "UP",
                "volume_climax": True,
                "etf_short_eligible": False,
            }

        # Check for exhaustion SELL (retail capitulation at top)
        cap_sell, cap_sell_conf, cap_sell_details = self.volume_detector.detect_exhaustion_sell(df, rsi, current_price)
        if cap_sell and cap_sell_conf >= 0.60:
            self.last_signal = "SELL"
            return {
                "signal": "SELL",
                "reason": (
                    f"[VOLUME CLIMAX] {self.asset} EXHAUSTION SELL\n"
                    f"  RSI={rsi:.1f} (deep overbought)\n"
                    f"  Volume={cap_sell_details.get('volume_ratio', 0):.1f}x avg (exhaustion spike)\n"
                    f"  Upper Wick={cap_sell_details.get('upper_wick_pct', 0):.1f}% (rejection of higher prices)\n"
                    f"  Confidence={cap_sell_conf:.2f} ✅"
                ),
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price,
                "confidence": cap_sell_conf,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "trend": "DOWN",
                "volume_climax": True,
                "etf_short_eligible": etf_short_eligible,
            }

        # ========== FALLBACK TO MA-BASED SIGNALS ==========
        # ---- BUY Signal: oversold + uptrend --------------------------------
        if rsi < self.rsi_oversold and fast_ma > slow_ma and macd.get("histogram", 0) > 0:
            confidence = self._bull_confidence(rsi, macd, fast_ma, slow_ma, sr, current_price)
            self.last_signal = "BUY"
            return {
                "signal": "BUY",
                "reason": (
                    f"OVERSOLD: RSI={rsi:.1f}, "
                    f"price near support ${sr.get('support', 0):.4f}"
                ),
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price,
                "confidence": confidence,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "trend": "UP",
                "volume_climax": False,
                "etf_short_eligible": False,
            }

        # ---- SELL Signal: overbought + downtrend ---------------------------
        current_position = self.position_manager.get_total_size()
        is_overbought = rsi > self.rsi_overbought
        is_downtrend = fast_ma < slow_ma
        macd_negative = macd.get("histogram", 0) < 0

        if is_overbought and is_downtrend and macd_negative:
            confidence = self._bear_confidence(rsi, macd, fast_ma, slow_ma, sr, current_price)

            # Build a human-readable reason that reflects ETF eligibility
            if not current_position and not etf_short_eligible:
                # No position to sell and ETF shorts unavailable — hold
                reason = (
                    f"OVERBOUGHT: RSI={rsi:.1f} but no open spot position "
                    f"and ETF shorts {'disabled' if not _ENABLE_SHORT_ETF else 'outside market hours'}"
                )
                signal = "HOLD"
            elif etf_short_eligible:
                reason = (
                    f"OVERBOUGHT: RSI={rsi:.1f}, "
                    f"price near resistance ${sr.get('resistance', 0):.4f} — "
                    f"SELL spot + rotate to ETHD/SETH"
                )
                signal = "SELL"
            else:
                reason = (
                    f"OVERBOUGHT: RSI={rsi:.1f}, "
                    f"price near resistance ${sr.get('resistance', 0):.4f} — "
                    f"SELL spot (ETF shorts {'disabled' if not _ENABLE_SHORT_ETF else 'outside market hours'})"
                )
                signal = "SELL" if current_position > 0 else "HOLD"

            self.last_signal = signal
            return {
                "signal": signal,
                "reason": reason,
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price,
                "confidence": confidence,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "trend": "DOWN",
                "volume_climax": False,
                "etf_short_eligible": etf_short_eligible,
                "is_bearish": True,
            }

        # ---- HOLD ----------------------------------------------------------
        trend_label = "UP" if fast_ma > slow_ma else "DOWN"
        self.last_signal = "HOLD"
        return {
            "signal": "HOLD",
            "reason": (
                f"Waiting for setup | RSI={rsi:.1f} | MA={trend_label}"
            ),
            "rsi": rsi,
            "macd": macd,
            "sr": sr,
            "price": current_price,
            "confidence": 0.0,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "trend": trend_label,
            "volume_climax": False,
            "etf_short_eligible": etf_short_eligible,
        }
    def process_signal(self, signal_data: dict, capital: float) -> bool:
        """
        Execute trade based on signal.

        Parameters
        ----------
        signal_data : dict
            Output from calculate_signals().
        capital : float
            USD amount to allocate (from PortfolioManager.capital_per_trade()).

        Returns
        -------
        bool – True if a trade was placed.
        """
        if self.paper_trading:
            return False

        signal = signal_data.get("signal")
        price = signal_data.get("price")
        reason = signal_data.get("reason", "")

        if not price or price <= 0:
            return False

        if signal == "BUY":
            pos_size = capital / price
            if pos_size <= 0:
                logger.warning("[%s] BUY skipped: capital too small ($%.4f)", self.asset, capital)
                return False
            self.position_manager.open_position(price, pos_size)
            logger.info(
                "[%s] BUY %.8f @ $%.4f | capital_used=$%.2f | %s",
                self.asset, pos_size, price, capital, reason,
            )
            return True

        elif signal == "SELL":
            current_size = self.position_manager.get_total_size()
            if current_size > 0:
                pnl = self.position_manager.close_position(price, current_size)
                logger.info(
                    "[%s] SELL %.8f @ $%.4f | PnL=$%.2f | %s",
                    self.asset, current_size, price, pnl, reason,
                )
                return True
            # Even if no spot position, signal the caller so ETF rotation can happen
            logger.info(
                "[%s] SELL signal (no spot position to close) | %s",
                self.asset, reason,
            )
            return False

        return False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict | None:
        """Get position summary with signal metadata."""
        summary = self.position_manager.get_position_summary()
        if summary:
            summary["last_rsi"] = self.last_rsi
            summary["last_signal"] = self.last_signal
        return summary

