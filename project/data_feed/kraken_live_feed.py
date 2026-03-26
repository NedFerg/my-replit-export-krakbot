#!/usr/bin/env python3
"""
Kraken Live Data Feed - Hourly Candles for Altcoins
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class KrakenLiveFeed:
    def __init__(self, pairs=['XXRPZUSD', 'LINKUSD', 'SOLUSD', 'XXLMZUSD', 'AVAXUSD']):
        self.pairs = pairs
        self.base_url = "https://api.kraken.com/0/public"
        self.data_cache = {}
        self.last_fetch = {}
        
    def get_ohlc(self, pair, interval=60):
        """Fetch hourly OHLC data from Kraken API"""
        try:
            if pair in self.last_fetch:
                if (datetime.utcnow() - self.last_fetch[pair]).total_seconds() < 300:
                    logger.debug(f"Using cached data for {pair}")
                    return self.data_cache.get(pair)
            
            url = f"{self.base_url}/OHLC"
            params = {'pair': pair, 'interval': interval}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('error'):
                logger.error(f"Kraken API error for {pair}: {data['error']}")
                return None
            
            pair_data = data.get('result', {}).get(pair, [])
            
            if not pair_data:
                logger.warning(f"No data returned for {pair}")
                return None
            
            pair_data = pair_data[-300:]
            
            df = pd.DataFrame(pair_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df[['open', 'high', 'low', 'close', 'vwap', 'volume']] = df[['open', 'high', 'low', 'close', 'vwap', 'volume']].astype(float)
            
            self.data_cache[pair] = df
            self.last_fetch[pair] = datetime.utcnow()
            
            logger.info(f"✓ Fetched {len(df)} hourly candles for {pair}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return None
    
    def fetch_all_pairs(self, interval=60):
        """Fetch latest hourly data for all configured pairs"""
        logger.info(f"Fetching hourly OHLC data for {len(self.pairs)} pairs...")
        
        feed_data = {}
        for pair in self.pairs:
            df = self.get_ohlc(pair, interval=interval)
            if df is not None:
                feed_data[pair] = df
        
        logger.info(f"Successfully fetched {len(feed_data)}/{len(self.pairs)} pairs")
        return feed_data
    
    def get_cached_data(self, pair):
        """Get cached data for a pair"""
        return self.data_cache.get(pair)
