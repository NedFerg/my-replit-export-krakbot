#!/usr/bin/env python3
"""
Trading Bot Runner v2 - Enhanced Signal System
RSI + MACD + Support/Resistance
Manages existing positions + new bot trades
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.config.ma_strategy_config import (
    STRATEGY_PARAMS,
    ENABLE_PAPER_TRADING,
    ENABLE_LIVE_TRADING,
    LOG_LEVEL,
    TRADING_PAIRS,
    POSITION_SIZE_PCT,
)
from project.data_feed.kraken_live_feed import KrakenLiveFeed
from project_scripts.trading_bot_live import KrakenAPI
from project_scripts.trading_bot_live_v2 import EnhancedTradeBot

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class TradingBotRunnerV2:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.mode = "PAPER" if paper_trading else "LIVE"
        
        # Get real balances from Kraken
        kraken_api = KrakenAPI()
        self.kraken_api = kraken_api
        
        available_balance = 0
        if kraken_api.enabled:
            available_balance = kraken_api.get_account_balance() or 51.01
        else:
            available_balance = 51.01
        
        self.available_balance = available_balance
        
        # Get all holdings
        self.all_balances = {}
        if kraken_api.enabled:
            self.all_balances = kraken_api.get_all_balances()
        
        # Mapping: Kraken internal code -> display name -> pair
        self.asset_map = {
            'XXRP': {'name': 'XRP', 'pair': 'XXRPZUSD'},
            'XXLM': {'name': 'XLM', 'pair': 'XXLMZUSD'},
            'SOL': {'name': 'SOL', 'pair': 'SOLUSD'},
            'AVAX': {'name': 'AVAX', 'pair': 'AVAXUSD'},
            'HBAR': {'name': 'HBAR', 'pair': 'HBARUSD'},
            'LINK': {'name': 'LINK', 'pair': 'LINKUSD'},
        }
        
        # Fetch current prices
        self.current_prices = self._fetch_all_prices()
        
        # Use pairs from config
        self.feed = KrakenLiveFeed(pairs=TRADING_PAIRS)
        
        # Initialize enhanced bots with existing positions
        self.bots = {}
        for kraken_code, info in self.asset_map.items():
            kraken_pair = info['pair']
            asset_name = info['name']
            
            # Check if this pair is in trading pairs
            if kraken_pair not in TRADING_PAIRS:
                continue
            
            # Get existing position
            existing_amount = float(self.all_balances.get(kraken_code, 0))
            current_price = self.current_prices.get(kraken_pair, 0)
            
            try:
                bot = EnhancedTradeBot(
                    asset_name=asset_name,
                    kraken_pair=kraken_pair,
                    existing_amount=existing_amount,
                    current_price=current_price,
                    paper_trading=paper_trading
                )
                self.bots[asset_name] = bot
            except Exception as e:
                logger.error(f"Failed to create bot for {asset_name}: {e}")
        
        # Log initialization
        self._log_initialization()
    
    def _fetch_all_prices(self):
        """Fetch current prices for all assets"""
        prices = {}
        for kraken_code, info in self.asset_map.items():
            pair = info['pair']
            try:
                response = requests.get(
                    "https://api.kraken.com/0/public/Ticker",
                    params={'pair': pair},
                    timeout=5
                )
                result = response.json()
                if result.get('result'):
                    ticker = list(result['result'].values())[0]
                    prices[pair] = float(ticker['c'][0])
            except:
                pass
        return prices
    
    def _log_initialization(self):
        """Log portfolio initialization"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("[PORTFOLIO INITIALIZATION - ENHANCED BOT v2]")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Available Cash (USD): ${self.available_balance:.2f}")
        logger.info("")
        logger.info("[TRADING BOTS] (with position tracking)")
        
        total_existing_value = 0
        for asset_name, bot in sorted(self.bots.items()):
            summary = bot.get_summary()
            if summary:
                logger.info(f"  {asset_name}: {summary['total_size']:.8f} @ ${summary['avg_entry_price']:.4f} (manual)")
                total_existing_value += summary['unrealized_pnl'] + (summary['avg_entry_price'] * summary['total_size'])
        
        logger.info("")
        logger.info(f"Total Existing Holdings Value: ${total_existing_value:.2f}")
        logger.info(f"Trading Capital Available: ${self.available_balance:.2f}")
        logger.info("=" * 80)
        logger.info("")
    
    def run(self):
        """Main trading loop"""
        iteration = 0
        
        while True:
            iteration += 1
            
            try:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f">>> ITERATION {iteration} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info("=" * 80)
                
                # Fetch data for all pairs
                feed_data = self.feed.fetch_all_pairs(interval=60)
                
                # Update prices and process signals
                logger.info("")
                logger.info("[SIGNAL ANALYSIS]")
                for asset_name, bot in sorted(self.bots.items()):
                    kraken_pair = bot.pair
                    
                    if kraken_pair in feed_data:
                        df = feed_data[kraken_pair]
                        if len(df) > 0:
                            # Calculate signals
                            signal_data = bot.calculate_signals(df)
                            signal = signal_data['signal']
                            reason = signal_data['reason']
                            
                            # Log signal
                            logger.info(f"  [{asset_name:6s}] {signal:5s} | {reason}")
                            
                            # Process signal
                            bot.process_signal(signal_data, self.available_balance * 0.1)
                
                # Portfolio summary
                logger.info("")
                logger.info("=" * 80)
                logger.info("[PORTFOLIO STATUS]")
                logger.info("=" * 80)
                
                total_unrealized = 0
                total_realized = 0
                
                for asset_name, bot in sorted(self.bots.items()):
                    summary = bot.get_summary()
                    if summary:
                        logger.info(f"  {asset_name:6s}:")
                        logger.info(f"    Position: {summary['total_size']:.8f}")
                        logger.info(f"    Entry:    ${summary['avg_entry_price']:.4f}")
                        logger.info(f"    Current:  ${summary['current_price']:.4f}")
                        logger.info(f"    Unrealized PnL: ${summary['unrealized_pnl']:+.2f} ({summary['unrealized_pnl_pct']:+.2f}%)")
                        logger.info(f"    Realized PnL:   ${summary['realized_pnl']:+.2f}")
                        total_unrealized += summary['unrealized_pnl']
                        total_realized += summary['realized_pnl']
                
                logger.info("")
                logger.info(f"Total Unrealized PnL: ${total_unrealized:+.2f}")
                logger.info(f"Total Realized PnL:   ${total_realized:+.2f}")
                logger.info(f"Available Cash:       ${self.available_balance:+.2f}")
                logger.info("=" * 80)
                
                # Wait for next cycle
                interval_seconds = 300
                logger.info(f"Next update in {interval_seconds}s...")
                logger.info("")
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("")
                logger.info("Bot interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

if __name__ == '__main__':
    logger.info("")
    logger.info("=" * 80)
    logger.info("[ENHANCED TRADING BOT v2] STARTED")
    logger.info(f"Mode: {'PAPER TRADING' if ENABLE_PAPER_TRADING else 'LIVE TRADING'}")
    logger.info("Signals: RSI + MACD + Support/Resistance")
    logger.info("=" * 80)
    
    runner = TradingBotRunnerV2(paper_trading=not ENABLE_LIVE_TRADING)
    runner.run()

