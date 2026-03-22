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
    
    def calculate_signals(self, df):
        """Calculate all trading signals"""
        if len(df) < 50:
            return {"status": "WAIT", "reason": "Insufficient data"}
        
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
        
        # BUY Signal: Oversold + Uptrend
        if rsi < self.rsi_oversold and fast_ma > slow_ma and macd['histogram'] > 0:
            return {
                "signal": "BUY",
                "reason": f"OVERSOLD: RSI={rsi:.1f}, Price near support ${sr['support']:.4f}",
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price
            }
        
        # SELL Signal: Overbought + Downtrend
        current_position = self.position_manager.get_total_size()
        if current_position > 0 and rsi > self.rsi_overbought and fast_ma < slow_ma and macd['histogram'] < 0:
            return {
                "signal": "SELL",
                "reason": f"OVERBOUGHT: RSI={rsi:.1f}, Price near resistance ${sr['resistance']:.4f}",
                "rsi": rsi,
                "macd": macd,
                "sr": sr,
                "atr": atr,
                "price": current_price
            }
        
        # HOLD signal
        return {
            "signal": "HOLD",
            "reason": f"Waiting for setup | RSI={rsi:.1f} | MA={'UP' if fast_ma > slow_ma else 'DOWN'}",
            "rsi": rsi,
            "macd": macd,
            "sr": sr,
            "price": current_price
        }
    
    def process_signal(self, signal_data, capital):
        """Execute trade based on signal"""
        if self.paper_trading:
            return False
        
        signal = signal_data.get('signal')
        price = signal_data.get('price')
        reason = signal_data.get('reason')
        
        if signal == "BUY":
            # Calculate position size
            pos_size = min(capital / price, self.max_position_size)
            self.position_manager.open_position(price, pos_size)
            logger.info(f"[{self.asset}] BUY: {pos_size:.8f} @ ${price:.4f} - {reason}")
            return True
        
        elif signal == "SELL":
            # Sell entire position
            current_size = self.position_manager.get_total_size()
            pnl = self.position_manager.close_position(price, current_size)
            logger.info(f"[{self.asset}] SELL: {current_size:.8f} @ ${price:.4f} - PnL=${pnl:.2f} - {reason}")
            return True
        
        return False
    
    def get_summary(self):
        """Get position summary"""
        summary = self.position_manager.get_position_summary()
        if summary:
            summary['last_rsi'] = self.last_rsi
            summary['last_signal'] = self.last_signal
        return summary

