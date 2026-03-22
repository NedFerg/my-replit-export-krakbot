#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from project.config.ma_strategy_config import STRATEGY_PARAMS, INITIAL_CAPITAL_USD
from project_scripts.trading_bot_live import MATradeBot

logging.basicConfig(level=logging.CRITICAL)

DATA_DIR = ROOT / "data" / "historical"
PAIRS = {
    'BTC': 'XXBTZUSD_1d.csv',
    'ETH': 'XETHZUSD_1d.csv',
    'XRP': 'XXRPZUSD_1d.csv',
    'LINK': 'LINKUSD_1d.csv',
}

results = []
for pair_name, csv_file in PAIRS.items():
    csv_path = DATA_DIR / csv_file
    if not csv_path.exists():
        continue
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    bot = MATradeBot(pair_name, paper_trading=True)
    
    for idx in range(bot.params['slow_ma'] + 5, len(df)):
        df_subset = df.iloc[:idx+1].copy()
        signal_info = bot.update_signal(df_subset)
        if not signal_info:
            continue
        current_price = signal_info['current_price']
        signal = signal_info['signal']
        if signal == 'BUY' and not bot.position:
            bot.execute_entry(current_price)
        elif signal == 'SELL' and bot.position:
            bot.execute_exit(current_price)
        bot.check_stop_loss(current_price)
        bot.check_take_profit(current_price)
    
    summary = bot.get_summary()
    results.append({
        'pair': pair_name,
        'trades': summary['trades'],
        'win_rate': summary.get('win_rate', 0),
        'capital': summary['capital'],
        'pnl_pct': summary.get('total_pnl_pct', 0),
    })

print("\nLIVE BOT BACKTEST RESULTS\n")
for r in results:
    print(f"{r['pair']:4} | {r['pnl_pct']:+6.1f}% | {r['trades']:2} trades | {r['win_rate']:4.0f}% win")

total_capital = sum(r['capital'] for r in results)
portfolio_pnl = (total_capital - INITIAL_CAPITAL_USD * len(results)) / (INITIAL_CAPITAL_USD * len(results)) * 100

print(f"\nPortfolio: ${total_capital:,.0f} | {portfolio_pnl:+.1f}%\n")
