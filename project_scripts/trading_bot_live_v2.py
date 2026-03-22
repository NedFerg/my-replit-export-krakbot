#!/usr/bin/env python3
"""
Enhanced Trading Bot v2 - Mean Reversion Swing Trader
Uses RSI + MACD + Support/Resistance for entries/exits
Tracks existing positions and pyramids into them
"""

import logging
from datetime import datetime
import pandas as pd

from project.config.ma_strategy_config import STRATEGY_PARAMS, INITIAL_CAPITAL_USD
from project_scripts.technical_indicators import TechnicalIndicators
from project_scripts.position_manager import PositionManager, Position

logger = logging.getLogger(__name__)

class EnhancedTradeBot:
    """Mean reversion swing trading bot"""
    
    def __init__(self, asset_name, kraken_pair, existing_amount=0, current_price=0, paper_trading=True):
        self.asset = asset_name
        self.pair = kraken_pair
        self.paper_trading = paper_trading
        self.params = STRATEGY_PARAMS.get(kraken_pair, {})
        
        # Position management
        self.position_manager = PositionManager(
            asset_name=asset_name,
            kraken_code=kraken_pair,
            existing_amount=existing_amount,
            current_price=current_price
        )
        
        # Trading parameters
        self.capital_per_trade = INITIAL_CAPITAL_USD
        self.max_position_size = existing_amount * 2  # Max 2x the existing amount
        self.stop_loss_pct = 5.0
        self.take_profit_pct = 15.0
        
        # Signal thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_signal_threshold = 0.0001
        
        # State tracking
        self.last_signal = "WAIT"
        self.last_rsi = 50
        self.last_macd = {}
        self.entry_history = []
        
        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(f"[{mode}] {asset_name} bot initialized | Existing: {existing_amount:.8f}")
    
    def _macd_strength(self, macd: dict) -> float:
        """
        Return a 0-1 score representing how strong/decisive the MACD histogram
        is relative to the MACD line magnitude.  Clamped to [0, 1].
        """
        _EPSILON = 1e-9
        hist = macd.get('histogram', 0)
        denominator = max(abs(macd.get('macd', 0)), _EPSILON)
        return min(abs(hist) / denominator, 1.0)

    def _bullish_confidence(self, rsi, macd, fast_ma, slow_ma, sr, price):
        """
        Score BUY signal quality 0-100.

        Scoring breakdown (each component adds up to its cap):
          - RSI depth below oversold threshold   → up to 40 pts
          - MACD histogram magnitude             → up to 30 pts
          - Price proximity to support           → up to 30 pts
        """
        score = 0

        # RSI component: deeper oversold → higher score
        rsi_headroom = max(self.rsi_oversold - rsi, 0)
        score += min(rsi_headroom / self.rsi_oversold * 40, 40)

        # MACD component: larger histogram → more momentum
        if macd.get('histogram', 0) > 0:
            score += self._macd_strength(macd) * 30

        # Price-to-support proximity
        if sr.get('support', 0) > 0 and price > 0:
            proximity = max(1 - (price - sr['support']) / price, 0)
            score += min(proximity * 30, 30)

        # MA alignment bonus
        if fast_ma > slow_ma:
            score = min(score * 1.1, 100)

        return round(score, 1)

    def _bearish_confidence(self, rsi, macd, fast_ma, slow_ma, sr, price):
        """
        Score SELL signal quality 0-100.

        Scoring breakdown:
          - RSI height above overbought threshold → up to 40 pts
          - MACD histogram depth below zero       → up to 30 pts
          - Price proximity to resistance          → up to 30 pts
        """
        score = 0

        # RSI component: deeper overbought → higher score
        rsi_excess = max(rsi - self.rsi_overbought, 0)
        rsi_range = 100 - self.rsi_overbought
        score += min(rsi_excess / rsi_range * 40, 40)

        # MACD component: more negative histogram → stronger downtrend
        if macd.get('histogram', 0) < 0:
            score += self._macd_strength(macd) * 30

        # Price-to-resistance proximity
        if sr.get('resistance', 0) > 0 and price > 0:
            proximity = max(1 - (sr['resistance'] - price) / price, 0)
            score += min(proximity * 30, 30)

        # MA alignment bonus
        if fast_ma < slow_ma:
            score = min(score * 1.1, 100)

        return round(score, 1)

    def calculate_signals(self, df):
        """Calculate all trading signals"""
        if len(df) < 50:
            return {"signal": "WAIT", "reason": "Insufficient data"}
        
        # Calculate technical indicators
        rsi = TechnicalIndicators.calculate_rsi(df, period=14)
        macd = TechnicalIndicators.calculate_macd(df)
        sr = TechnicalIndicators.find_support_resistance(df, lookback=20)
        atr = TechnicalIndicators.calculate_atr(df, period=14)
        
        # MA signals
        fast_ma = df['close'].tail(self.params.get('fast_ma', 5)).mean()
        slow_ma = df['close'].tail(self.params.get('slow_ma', 100)).mean()
        current_price = df['close'].iloc[-1]
        
        self.last_rsi = rsi
        self.last_macd = macd
        
        # BUY Signal: Oversold + Uptrend + MACD positive
        if rsi < self.rsi_oversold and fast_ma > slow_ma and macd['histogram'] > 0:
            confidence = self._bullish_confidence(rsi, macd, fast_ma, slow_ma, sr, current_price)
            return {
                "signal": "BUY",
                "reason": (
                    f"OVERSOLD: RSI={rsi:.1f}, Price near support "
                    f"${sr['support']:.4f}, confidence={confidence:.0f}/100"
                ),
                "confidence": confidence,
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "trend": "UP",
            }
        
        # SELL Signal: Overbought + Downtrend + MACD negative
        current_position = self.position_manager.get_total_size()
        if current_position > 0 and rsi > self.rsi_overbought and fast_ma < slow_ma and macd['histogram'] < 0:
            confidence = self._bearish_confidence(rsi, macd, fast_ma, slow_ma, sr, current_price)
            return {
                "signal": "SELL",
                "reason": (
                    f"OVERBOUGHT: RSI={rsi:.1f}, Price near resistance "
                    f"${sr['resistance']:.4f}, confidence={confidence:.0f}/100"
                ),
                "confidence": confidence,
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price,
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "trend": "DOWN",
                "position_size": current_position,
            }
        
        # HOLD signal
        return {
            "signal": "HOLD",
            "reason": f"Waiting for setup | RSI={rsi:.1f} | MA={'UP' if fast_ma > slow_ma else 'DOWN'}",
            "rsi": rsi,
            "macd": macd,
            "sr": sr,
            "price": current_price,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "trend": "UP" if fast_ma > slow_ma else "DOWN",
        }
    
    def process_signal(self, signal_data, capital):
        """
        Execute trade based on signal.

        Returns
        -------
        (executed: bool, realized_pnl: float)
            executed     – True when an order was placed.
            realized_pnl – Realized PnL in USD for SELL orders (0 for BUY/HOLD).
        """
        if self.paper_trading:
            return False, 0.0
        
        signal = signal_data.get('signal')
        price = signal_data.get('price')
        reason = signal_data.get('reason')
        
        if signal == "BUY":
            # Calculate position size
            if not price:
                logger.warning(f"[{self.asset}] BUY signal ignored: price is zero or missing")
                return False, 0.0
            pos_size = min(capital / price, self.max_position_size) if price else 0
            if pos_size <= 0:
                return False, 0.0
            self.position_manager.open_position(price, pos_size)
            self.last_signal = "BUY"
            logger.info(f"[{self.asset}] BUY: {pos_size:.8f} @ ${price:.4f} - {reason}")
            return True, 0.0
        
        elif signal == "SELL":
            # Sell entire position and capture realized PnL
            current_size = self.position_manager.get_total_size()
            if current_size <= 0:
                return False, 0.0
            pnl = self.position_manager.close_position(price, current_size)
            self.last_signal = "SELL"
            logger.info(
                f"[{self.asset}] SELL: {current_size:.8f} @ ${price:.4f} "
                f"- PnL=${pnl:.2f} - {reason}"
            )
            return True, pnl
        
        return False, 0.0
    
    def get_summary(self):
        """Get position summary"""
        summary = self.position_manager.get_position_summary()
        if summary:
            summary['last_rsi'] = self.last_rsi
            summary['last_signal'] = self.last_signal
        return summary

