#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "historical"

PAIRS = {
    'BTC': ('XXBTZUSD_1d.csv', 5, 100),      # MA(5,100)
    'ETH': ('XETHZUSD_1d.csv', 5, 100),      # MA(5,100)
    'XRP': ('XXRPZUSD_1d.csv', 12, 100),     # MA(12,100)
    'LINK': ('LINKUSD_1d.csv', 12, 100),     # MA(12,100)
}

def run_ma_backtest(pair_name, csv_file, fast_ma, slow_ma):
    """MA crossover with optimized parameters"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Testing %s with MA(%d,%d)", pair_name, fast_ma, slow_ma)
    logger.info("=" * 70)
    
    csv_path = DATA_DIR / csv_file
    if not csv_path.exists():
        logger.error("❌ File not found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info("Data: %s to %s (%d candles)",
                   df['timestamp'].min().date(),
                   df['timestamp'].max().date(),
                   len(df))
        
        # Calculate MAs
        df['MA_fast'] = df['close'].rolling(fast_ma).mean()
        df['MA_slow'] = df['close'].rolling(slow_ma).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['MA_fast'] > df['MA_slow'], 'signal'] = 1
        df.loc[df['MA_fast'] < df['MA_slow'], 'signal'] = -1
        
        # Simulate trades
        initial_capital = 1000
        equity = initial_capital
        position = 0
        entry_price = 0
        trades = []
        
        for idx, row in df.iterrows():
            if row['signal'] == 1 and position == 0:
                position = 1
                entry_price = row['close']
            elif row['signal'] == -1 and position == 1:
                pnl_pct = (row['close'] - entry_price) / entry_price * 100
                equity *= (1 + pnl_pct / 100)
                trades.append({'pnl_pct': pnl_pct})
                position = 0
        
        total_return_pct = (equity - initial_capital) / initial_capital * 100
        win_rate = len([t for t in trades if t['pnl_pct'] > 0]) / len(trades) * 100 if trades else 0
        
        logger.info("✓ Return: %+.2f%% | Trades: %d | Win Rate: %.1f%%",
                   total_return_pct, len(trades), win_rate)
        logger.info("✓ Final Equity: $%.2f", equity)
        
        return {
            'pair': pair_name,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'return': total_return_pct,
            'trades': len(trades),
            'win_rate': win_rate,
            'equity': equity
        }
        
    except Exception as e:
        logger.error("❌ Error: %s", e)
        return None

def main():
    logger.info("=" * 70)
    logger.info("OPTIMIZED MA STRATEGY BACKTEST")
    logger.info("=" * 70)
    
    results = []
    for pair_name, (csv_file, fast_ma, slow_ma) in PAIRS.items():
        result = run_ma_backtest(pair_name, csv_file, fast_ma, slow_ma)
        if result:
            results.append(result)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    total_return = 0
    for r in results:
        logger.info("%s MA(%d,%d): %+7.2f%% | %2d trades | %5.1f%% win rate",
                   r['pair'], r['fast_ma'], r['slow_ma'], 
                   r['return'], r['trades'], r['win_rate'])
        total_return += r['return']
    
    avg_return = total_return / len(results) if results else 0
    logger.info("-" * 70)
    logger.info("Portfolio Average Return: %+.2f%%", avg_return)
    logger.info("=" * 70)

if __name__ == '__main__':
    main()
