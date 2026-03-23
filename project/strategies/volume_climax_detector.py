"""Volume capitulation and exhaustion pattern detector."""

class VolumeClimaxDetector:
    """
    Detect volume capitulation and exhaustion patterns.
    
    BUY pattern: RSI oversold + volume spike + long lower wick
    SELL pattern: RSI overbought + volume spike + long upper wick
    """
    
    def __init__(self, 
                 volume_spike_multiplier=2.5,
                 wick_ratio_threshold=0.03,
                 lookback_periods=50):
        self.volume_spike_multiplier = volume_spike_multiplier
        self.wick_ratio_threshold = wick_ratio_threshold
        self.lookback_periods = lookback_periods
    
    def detect_capitulation_buy(self, df, rsi, current_price):
        """Detect capitulation buy (institutional buyers stepping in)."""
        if len(df) < self.lookback_periods:
            return False, 0.0, {}
        
        if rsi >= 30:
            return False, 0.0, {"reason": "RSI not oversold"}
        
        current_vol = float(df['volume'].iloc[-1])
        avg_vol = float(df['volume'].tail(self.lookback_periods).mean())
        
        if avg_vol == 0:
            return False, 0.0, {"reason": "No volume data"}
        
        vol_ratio = current_vol / avg_vol
        if vol_ratio < self.volume_spike_multiplier:
            return False, 0.0, {"reason": f"Volume only {vol_ratio:.1f}x avg"}
        
        current_candle = df.iloc[-1]
        high = float(current_candle['high'])
        low = float(current_candle['low'])
        close = float(current_candle['close'])
        open_price = float(current_candle['open'])
        
        candle_height = high - low
        if candle_height == 0:
            return False, 0.0, {"reason": "Zero candle height"}
        
        body_bottom = min(open_price, close)
        lower_wick = body_bottom - low
        wick_ratio = lower_wick / candle_height if candle_height > 0 else 0
        
        if wick_ratio < self.wick_ratio_threshold:
            return False, 0.0, {"reason": f"Lower wick ratio only {wick_ratio:.3f}"}
        
        confidence = min(
            0.5 +
            0.2 * (rsi / 30) +
            0.2 * min(vol_ratio / (self.volume_spike_multiplier * 2), 1.0) +
            0.1 * min(wick_ratio / 0.05, 1.0),
            1.0
        )
        
        details = {
            "rsi": rsi,
            "volume_ratio": vol_ratio,
            "wick_ratio": wick_ratio,
            "lower_wick_pct": (lower_wick / close) * 100 if close > 0 else 0,
        }
        
        return True, confidence, details
    
    def detect_exhaustion_sell(self, df, rsi, current_price):
        """Detect exhaustion sell (retail capitulation at top)."""
        if len(df) < self.lookback_periods:
            return False, 0.0, {}
        
        if rsi <= 70:
            return False, 0.0, {"reason": "RSI not overbought"}
        
        current_vol = float(df['volume'].iloc[-1])
        avg_vol = float(df['volume'].tail(self.lookback_periods).mean())
        
        if avg_vol == 0:
            return False, 0.0, {"reason": "No volume data"}
        
        vol_ratio = current_vol / avg_vol
        if vol_ratio < self.volume_spike_multiplier:
            return False, 0.0, {"reason": f"Volume only {vol_ratio:.1f}x avg"}
        
        current_candle = df.iloc[-1]
        high = float(current_candle['high'])
        low = float(current_candle['low'])
        close = float(current_candle['close'])
        open_price = float(current_candle['open'])
        
        candle_height = high - low
        if candle_height == 0:
            return False, 0.0, {"reason": "Zero candle height"}
        
        body_top = max(open_price, close)
        upper_wick = high - body_top
        wick_ratio = upper_wick / candle_height if candle_height > 0 else 0
        
        if wick_ratio < self.wick_ratio_threshold:
            return False, 0.0, {"reason": f"Upper wick ratio only {wick_ratio:.3f}"}
        
        confidence = min(
            0.5 +
            0.2 * ((100 - rsi) / 30) +
            0.2 * min(vol_ratio / (self.volume_spike_multiplier * 2), 1.0) +
            0.1 * min(wick_ratio / 0.05, 1.0),
            1.0
        )
        
        details = {
            "rsi": rsi,
            "volume_ratio": vol_ratio,
            "wick_ratio": wick_ratio,
            "upper_wick_pct": (upper_wick / close) * 100 if close > 0 else 0,
        }
        
        return True, confidence, details
