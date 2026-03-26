#!/usr/bin/env python3
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = {
    'BTC': 'XXBTZUSD',
    'ETH': 'XETHZUSD',
    'XRP': 'XXRPZUSD',
    'SOL': 'SOL4ZUSD',
    'LINK': 'LINKUSD'
}

START_DATE = datetime(2020, 9, 1)
END_DATE = datetime(2022, 9, 30)

def fetch_kraken_ohlc(pair, interval=1440):
    """Fetch OHLC from Kraken (interval in minutes: 1=1m, 5=5m, 1440=1d)"""
    print(f"\n📥 Fetching {pair}...")
    
    url = "https://api.kraken.com/0/public/OHLC"
    
    all_data = []
    since = int(START_DATE.timestamp())
    end_ts = int(END_DATE.timestamp())
    
    while since < end_ts:
        params = {
            'pair': pair,
            'interval': interval,
            'since': since
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('error'):
                print(f"  ⚠️  Kraken error: {data['error']}")
                break
            
            ohlc = data.get('result', {}).get(pair, [])
            if not ohlc:
                break
            
            all_data.extend(ohlc)
            last_ts = ohlc[-1][0]
            since = last_ts
            
            print(f"  ✓ Fetched {len(ohlc)} candles (last: {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d')})")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            break
    
    if not all_data:
        print(f"❌ No data for {pair}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
        'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
    })
    
    output_file = DATA_DIR / f"{pair}_1d.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df):,} candles → {output_file}")
    
    return df

def main():
    print("=" * 70)
    print("KRAKEN PUBLIC API HISTORICAL DATA FETCHER")
    print("=" * 70)
    print(f"Pairs: {', '.join(SYMBOLS.keys())}")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print("Interval: Daily (1440-min)")
    print("=" * 70)
    
    total_candles = 0
    for code, pair in SYMBOLS.items():
        df = fetch_kraken_ohlc(pair, interval=1440)
        if df is not None:
            total_candles += len(df)
    
    print("\n" + "=" * 70)
    print(f"✓ Done! Total candles: {total_candles:,}")
    print("=" * 70)

if __name__ == '__main__':
    main()
