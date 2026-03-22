#!/usr/bin/env python3
"""
Trading Executor - Handles BUY/SELL orders and position management
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, kraken_api, bot_name, kraken_pair):
        self.kraken = kraken_api
        self.bot_name = bot_name
        self.kraken_pair = kraken_pair
        self.position = None
        self.entry_price = None
        self.entry_time = None
    
    def execute_buy(self, price, volume, capital):
        """Execute buy order"""
        if self.position is not None:
            logger.warning(f"[{self.bot_name}] Already in position, skipping BUY")
            return False
        
        try:
            # Calculate volume to buy
            buy_volume = (capital * 0.95) / price  # Use 95% of capital
            
            logger.info(f"[{self.bot_name}] 🟢 BUY SIGNAL")
            logger.info(f"  Price: ${price:.4f}")
            logger.info(f"  Volume: {buy_volume:.4f}")
            logger.info(f"  Capital: ${capital:.2f}")
            
            # Place order
            order_id = self.kraken.place_order(
                pair=self.kraken_pair,
                side='buy',
                volume=buy_volume,
                price=price
            )
            
            if order_id:
                self.position = {
                    'side': 'buy',
                    'volume': buy_volume,
                    'entry_price': price,
                    'entry_time': datetime.utcnow(),
                    'order_id': order_id
                }
                self.entry_price = price
                self.entry_time = datetime.utcnow()
                logger.info(f"[{self.bot_name}] ✓ Buy order placed: {order_id}")
                return True
            else:
                logger.error(f"[{self.bot_name}] Failed to place buy order")
                return False
                
        except Exception as e:
            logger.error(f"[{self.bot_name}] Error executing buy: {e}")
            return False
    
    def execute_sell(self, price, reason="SIGNAL"):
        """Execute sell order"""
        if self.position is None:
            logger.warning(f"[{self.bot_name}] No position to sell, skipping SELL")
            return False
        
        try:
            volume = self.position['volume']
            entry_price = self.position['entry_price']
            
            pnl = ((price - entry_price) / entry_price) * 100
            profit = (price - entry_price) * volume
            
            logger.info(f"[{self.bot_name}] 🔴 SELL SIGNAL ({reason})")
            logger.info(f"  Entry: ${entry_price:.4f} | Exit: ${price:.4f}")
            logger.info(f"  PnL: {pnl:+.2f}% | Profit: ${profit:+.2f}")
            
            # Place sell order
            order_id = self.kraken.place_order(
                pair=self.kraken_pair,
                side='sell',
                volume=volume,
                price=price
            )
            
            if order_id:
                logger.info(f"[{self.bot_name}] ✓ Sell order placed: {order_id}")
                
                # Record trade
                trade = {
                    'entry_price': entry_price,
                    'exit_price': price,
                    'volume': volume,
                    'pnl_pct': pnl,
                    'pnl_usd': profit,
                    'entry_time': self.entry_time,
                    'exit_time': datetime.utcnow(),
                    'reason': reason
                }
                
                self.position = None
                self.entry_price = None
                self.entry_time = None
                
                return trade
            else:
                logger.error(f"[{self.bot_name}] Failed to place sell order")
                return False
                
        except Exception as e:
            logger.error(f"[{self.bot_name}] Error executing sell: {e}")
            return False
    
    def check_stop_loss(self, current_price, stop_loss_pct):
        """Check if stop loss is hit"""
        if self.position is None:
            return False
        
        loss_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        
        if loss_pct >= stop_loss_pct:
            logger.warning(f"[{self.bot_name}] ⛔ STOP LOSS HIT: {loss_pct:.2f}%")
            return True
        
        return False
    
    def check_take_profit(self, current_price, take_profit_pct):
        """Check if take profit is hit"""
        if self.position is None:
            return False
        
        gain_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        if gain_pct >= take_profit_pct:
            logger.info(f"[{self.bot_name}] 💰 TAKE PROFIT HIT: {gain_pct:.2f}%")
            return True
        
        return False
    
    def is_in_position(self):
        """Check if currently in position"""
        return self.position is not None
