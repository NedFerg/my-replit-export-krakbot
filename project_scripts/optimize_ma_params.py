#!/usr/bin/env python3
import logging
import sys
from pathlib import Path
import pandas as pd
import itertools

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
    'BTC': 'XXBTZUSD_1d.csv',
    'ETH': 'XETHZUSD_1d.csv',
    'XRP': 'XXRPZUSD_1d.csv',
    'LINK': 'LINKUSD_1d.csv',
}

# Test these MA combinations
FAST_MAs = [3, 5, 8, 10, 12]
SLOW_MAs = [20, 30, 50, 75, 100]

def run_ma_backtest(pair_name, csv_file, fast_ma, slow_ma):
    """MA crossover backtest with configurable periods"""
    csv_path = DATA_DIR / csv_file
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < slow_ma + 10:
            return None
        
        # Calculate MAs
        df['MA_fast'] = df['close'].rolling(fast_ma).mean()
        df['MA_slow'] = df['close'].rolling(slow_ma).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['MA_fast'] > df['MA_slow'], 'signal'] = 1   # BUY
        df.loc[df['MA_fast'] < df['MA_slow'], 'signal'] = -1  # SELL
        
        # Simulate trades
        initial_capital = 1000
        equity = initial_capital
        position = 0
        entry_price = 0
        trades = []
        
        for idx, row in df.iterrows():
            if row['signal'] == 1 and position == 0:  # BUY
                position = 1
                entry_price = row['close']
            elif row['signal'] == -1 and position == 1:  # SELL
                pnl_pct = (row['close'] - entry_price) / entry_price * 100
                equity *= (1 + pnl_pct / 100)
                trades.append({'pnl_pct': pnl_pct})
                position = 0
        
        total_return_pct = (equity - initial_capital) / initial_capital * 100
        win_rate = len([t for t in trades if t['pnl_pct'] > 0]) / len(trades) * 100 if trades else 0
        
        return {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'return': total_return_pct,
            'trades': len(trades),
            'win_rate': win_rate,
            'equity': equity
        }
        
    except Exception as e:
        return None

def main():
    logger.info("=" * 80)
    logger.info("MA PARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info("Testing Fast MA: %s | Slow MA: %s", FAST_MAs, SLOW_MAs)
    logger.info("=" * 80)
    
    for pair_name, csv_file in PAIRS.items():
        logger.info("")
        logger.info("=" * 80)
        logger.info("PAIR: %s", pair_name)
        logger.info("=" * 80)
        
        results = []
        for fast_ma, slow_ma in itertools.product(FAST_MAs, SLOW_MAs):
            result = run_ma_backtest(pair_name, csv_file, fast_ma, slow_ma)
            if result:
                results.append(result)
        
        # Sort by return (best first)
        results.sort(key=lambda x: x['return'], reverse=True)
        
        # Show top 5
        logger.info("")
        logger.info("TOP 5 PARAMETER COMBINATIONS:")
        logger.info("-" * 80)
        for i, r in enumerate(results[:5], 1):
            logger.info("%d. MA(%d,%d): %+7.2f%% | %2d trades | %5.1f%% win rate",
                       i, r['fast_ma'], r['slow_ma'], r['return'], 
                       r['trades'], r['win_rate'])
        
        # Show worst 3 (for comparison)
        logger.info("")
        logger.info("WORST 3 (for reference):")
        logger.info("-" * 80)
        for i, r in enumerate(results[-3:], 1):
            logger.info("%d. MA(%d,%d): %+7.2f%% | %2d trades | %5.1f%% win rate",
                       i, r['fast_ma'], r['slow_ma'], r['return'], 
                       r['trades'], r['win_rate'])

if __name__ == '__main__':
    main()
