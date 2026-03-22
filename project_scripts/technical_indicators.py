#!/usr/bin/env python3
"""
Technical Indicators for Trading Signals
RSI, MACD, Support/Resistance
"""

import numpy as np
import pandas as pd

class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """Calculate MACD and signal line"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        histogram = macd - macd_signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': macd_signal.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    @staticmethod
    def find_support_resistance(df, lookback=20):
        """Find support and resistance levels"""
        high = df['high'].tail(lookback).max()
        low = df['low'].tail(lookback).min()
        
        return {
            'resistance': high,
            'support': low,
            'midpoint': (high + low) / 2
        }
    
    @staticmethod
    def calculate_atr(df, period=14):
        """Calculate Average True Range for volatility"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(window=period).mean()
        return atr.iloc[-1] if len(atr) > 0 else 0

