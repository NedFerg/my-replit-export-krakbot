#!/usr/bin/env python3
"""
Backtest using the Live Trading Bot
Simulates the bot executing trades on historical data
"""

import logging
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.config.ma_strategy_config import (
    STRATEGY_PARAMS,
    INITIAL_CAPITAL_USD,
    LOG_LEVEL
)
from project_scripts.trading_bot_live import MATradeBot

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "historical"

PAIRS = {
    'BTC': 'XXBTZUSD_1d.csv',
    'ETH': 'XETHZUSD_1d.csv',
    'XRP': 'XXRPZUSD_1d.csv',
    'LINK': 'LINKUSD_1d.csv',
}

def run_backtest_with_bot(pair_name, csv_file):
    """Run backtest using the live bot logic"""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BACKTESTING {pair_name} with Live Trading Bot")
    logger.info("=" * 80)
    
    csv_path = DATA_DIR / csv_file
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data: {df['timestamp'].min().date()} to {df['timestamp'].max().date()} ({len(df)} candles)")
        
        # Initialize bot
        bot = MATradeBot(pair_name, paper_trading=True)
        
        # Simulate trading day by day
        for idx in range(bot.params['slow_ma'] + 5, len(df)):
            # Get data up to current candle
            df_subset = df.iloc[:idx+1].copy()
            
            # Get signal
            signal_info = bot.update_signal(df_subset)
            
            if not signal_info:
                continue
            
            current_price = signal_info['current_price']
            signal = signal_info['signal']
            timestamp = signal_info['timestamp']
            
            # Log every 50 candles
            if idx % 50 == 0:
                logger.info(f"[{timestamp.date()}] Signal: {signal} | Price: ${current_price:.2f}")
            
            # Execute trades
            if signal == 'BUY' and not bot.position:
                logger.info(f"[BUY] {timestamp.date()} @ ${current_price:.2f}")
                bot.execute_entry(current_price)
            
            elif signal == 'SELL' and bot.position:
                logger.info(f"[SELL] {timestamp.date()} @ ${current_price:.2f}")
                bot.execute_exit(current_price)
            
            # Check stops
            bot.check_stop_loss(current_price)
            bot.check_take_profit(current_price)
        
        # Get final summary
        summary = bot.get_summary()
        
        logger.info("")
        logger.info("-" * 80)
        logger.info(f"FINAL RESULTS for {pair_name}:")
        logger.info(f"  Trades: {summary['trades']}")
        logger.info(f"  Win Rate: {summary.get('win_rate', 0):.1f}%")
        logger.info(f"  Final Capital: ${summary['capital']:,.2f}")
        logger.info(f"  PnL: {summary.get('total_pnl_pct', 0):+.2f}%")
        logger.info("-" * 80)
        
        return {
            'pair': pair_name,
            'trades': summary['trades'],
            'win_rate': summary.get('win_rate', 0),
            'capital': summary['capital'],
            'pnl_pct': summary.get('total_pnl_pct', 0),
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None

def main():
    logger.info("")
    logger.info("=" * 80)
    logger.info("LIVE TRADING BOT BACKTEST - SIMULATED TRADING")
    logger.info("=" * 80)
    logger.info("Running bot logic on historical data to see it trade")
    logger.info("=" * 80)
    
    results = []
    for pair_name, csv_file in PAIRS.items():
        result = run_backtest_with_bot(pair_name, csv_file)
        if result:
            results.append(result)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PORTFOLIO SUMMARY")
    logger.info("=" * 80)
    
    total_capital = 0
    total_pnl = 0
    
    for r in results:
        total_capital += r['capital']
        total_pnl += (r['capital'] - INITIAL_CAPITAL_USD)
        logger.info(f"{r['pair']}: {r['pnl_pct']:+.2f}% | {r['trades']} trades | {r['win_rate']:.1f}% win")
    
    portfolio_pnl = total_pnl / (INITIAL_CAPITAL_USD * len(results)) * 100 if results else 0
    logger.info("-" * 80)
    logger.info(f"Total Capital: ${total_capital:,.2f}")
    logger.info(f"Portfolio PnL: {portfolio_pnl:+.2f}%")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
