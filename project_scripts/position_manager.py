#!/usr/bin/env python3
"""
Position Manager - Tracks all positions (manual + bot trades)
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Position:
    """Represents a single trading position"""
    
    def __init__(self, asset, entry_price, entry_size, entry_time=None, is_manual=False):
        self.asset = asset
        self.entry_price = entry_price
        self.entry_size = entry_size
        self.entry_time = entry_time or datetime.utcnow()
        self.is_manual = is_manual
        self.current_price = entry_price
        self.exit_price = None
        self.exit_time = None
        self.status = "OPEN"  # OPEN, CLOSED, PARTIAL
        self.trades = []  # For partial exits
    
    def update_price(self, current_price):
        """Update current market price"""
        self.current_price = current_price
    
    def get_unrealized_pnl(self):
        """Get unrealized PnL in USD"""
        return (self.current_price - self.entry_price) * self.entry_size
    
    def get_unrealized_pnl_pct(self):
        """Get unrealized PnL percentage"""
        if self.entry_price == 0:
            return 0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def close(self, exit_price, exit_size=None):
        """Close position or reduce size"""
        if exit_size is None:
            exit_size = self.entry_size
        
        realized_pnl = (exit_price - self.entry_price) * exit_size
        self.trades.append({
            'exit_price': exit_price,
            'exit_size': exit_size,
            'realized_pnl': realized_pnl,
            'exit_time': datetime.utcnow()
        })
        
        self.entry_size -= exit_size
        if self.entry_size <= 0:
            self.status = "CLOSED"
            self.exit_price = exit_price
            self.exit_time = datetime.utcnow()
        else:
            self.status = "PARTIAL"
        
        return realized_pnl
    
    def add_to_position(self, add_price, add_size):
        """Add to existing position (pyramid/average up)"""
        total_cost = (self.entry_price * self.entry_size) + (add_price * add_size)
        total_size = self.entry_size + add_size
        self.entry_price = total_cost / total_size
        self.entry_size = total_size


class PositionManager:
    """Manages all positions for an asset"""
    
    def __init__(self, asset_name, kraken_code, existing_amount=0, current_price=0):
        self.asset = asset_name
        self.kraken_code = kraken_code
        self.positions = []
        
        # Add existing position if any
        if existing_amount > 0:
            self.positions.append(
                Position(
                    asset=asset_name,
                    entry_price=current_price,
                    entry_size=existing_amount,
                    is_manual=True
                )
            )
            logger.info(f"[{asset_name}] Loaded existing position: {existing_amount:.8f} @ ${current_price:.4f}")
    
    def open_position(self, entry_price, entry_size):
        """Open new trading position"""
        pos = Position(
            asset=self.asset,
            entry_price=entry_price,
            entry_size=entry_size,
            is_manual=False
        )
        self.positions.append(pos)
        logger.info(f"[{self.asset}] Opened position: {entry_size:.8f} @ ${entry_price:.4f}")
        return pos
    
    def close_position(self, exit_price, exit_size=None):
        """Close oldest open position"""
        for pos in self.positions:
            if pos.status == "OPEN":
                pnl = pos.close(exit_price, exit_size)
                logger.info(f"[{self.asset}] Closed position: PnL=${pnl:.2f}")
                return pnl
        return 0
    
    def get_total_size(self):
        """Get total position size across all open positions"""
        return sum(pos.entry_size for pos in self.positions if pos.status in ["OPEN", "PARTIAL"])
    
    def get_total_unrealized_pnl(self):
        """Get total unrealized PnL"""
        return sum(pos.get_unrealized_pnl() for pos in self.positions if pos.status in ["OPEN", "PARTIAL"])
    
    def get_total_realized_pnl(self):
        """Get total realized PnL from closed trades"""
        total = 0
        for pos in self.positions:
            if pos.status == "CLOSED":
                for trade in pos.trades:
                    total += trade['realized_pnl']
        return total
    
    def update_prices(self, current_price):
        """Update all position prices"""
        for pos in self.positions:
            if pos.status in ["OPEN", "PARTIAL"]:
                pos.update_price(current_price)
    
    def get_position_summary(self):
        """Get summary of all positions"""
        open_positions = [p for p in self.positions if p.status in ["OPEN", "PARTIAL"]]
        if not open_positions:
            return None
        
        total_size = self.get_total_size()
        total_unrealized = self.get_total_unrealized_pnl()
        total_unrealized_pct = (total_unrealized / (open_positions[0].entry_price * total_size) * 100) if total_size > 0 else 0
        
        return {
            'asset': self.asset,
            'total_size': total_size,
            'avg_entry_price': sum(p.entry_price * p.entry_size for p in open_positions) / total_size if total_size > 0 else 0,
            'current_price': open_positions[0].current_price if open_positions else 0,
            'unrealized_pnl': total_unrealized,
            'unrealized_pnl_pct': total_unrealized_pct,
            'realized_pnl': self.get_total_realized_pnl(),
            'manual_position': any(p.is_manual for p in open_positions)
        }

