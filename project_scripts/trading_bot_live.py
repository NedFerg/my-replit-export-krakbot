#!/usr/bin/env python3
"""
Live Trading Bot with Kraken API Integration
"""

import logging
import os
import time
import hashlib
import hmac
import base64
import urllib.parse
import requests
from datetime import datetime

from project.config.ma_strategy_config import STRATEGY_PARAMS, INITIAL_CAPITAL_USD
from project_scripts.trading_executor import TradeExecutor

logger = logging.getLogger(__name__)

class KrakenAPI:
    def __init__(self):
        self.api_key = os.getenv('KRAKEN_API_KEY')
        self.api_secret = os.getenv('KRAKEN_API_SECRET')
        self.base_url = "https://api.kraken.com"
        
        if not self.api_key or not self.api_secret:
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✓ Kraken API credentials loaded")
    
    def _get_signature(self, urlpath, data):
        """Generate Kraken API signature"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512
        )
        return base64.b64encode(signature.digest()).decode()
    
    def get_account_balance(self):
        """Get USD balance"""
        if not self.enabled:
            return None
        
        try:
            urlpath = "/0/private/Balance"
            nonce = int(time.time() * 1000)
            data = {'nonce': nonce}
            
            headers = {
                'API-Sign': self._get_signature(urlpath, data),
                'API-Key': self.api_key,
            }
            
            response = requests.post(
                self.base_url + urlpath,
                headers=headers,
                data=data,
                timeout=10
            )
            
            result = response.json()
            if result.get('error'):
                logger.error(f"Kraken error: {result['error']}")
                return None
            
            balances = result.get('result', {})
            usd_balance = float(balances.get('ZUSD', 0))
            logger.info(f"✓ Account balance: ${usd_balance:.2f} USD available")
            return usd_balance
            
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return None
    
    def get_all_balances(self):
        """Get ALL balances (crypto + USD)"""
        if not self.enabled:
            return {}
        
        try:
            urlpath = "/0/private/Balance"
            nonce = int(time.time() * 1000)
            data = {'nonce': nonce}
            
            headers = {
                'API-Sign': self._get_signature(urlpath, data),
                'API-Key': self.api_key,
            }
            
            response = requests.post(
                self.base_url + urlpath,
                headers=headers,
                data=data,
                timeout=10
            )
            
            result = response.json()
            if result.get('error'):
                logger.error(f"Kraken error: {result['error']}")
                return {}
            
            return result.get('result', {})
            
        except Exception as e:
            logger.error(f"Error getting all balances: {e}")
            return {}

class MATradeBot:
    def __init__(self, pair, paper_trading=True):
        self.pair = pair
        self.paper_trading = paper_trading
        self.params = STRATEGY_PARAMS.get(pair)
        if not self.params:
            raise ValueError(f"Unknown pair: {pair}")
        
        self.capital = INITIAL_CAPITAL_USD
        self.trades = []
        self.kraken = KrakenAPI() if not paper_trading else None
        self.executor = None
        self.kraken_pair = None
        
        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(f"[{mode}] Initialized {pair} bot with MA({self.params['fast_ma']},{self.params['slow_ma']})")
    
    def get_signal(self, df):
        """Calculate MA signal"""
        if len(df) < self.params['slow_ma'] + 5:
            return "WAIT"
        
        fast_ma = df['close'].tail(self.params['fast_ma']).mean()
        slow_ma = df['close'].tail(self.params['slow_ma']).mean()
        
        if fast_ma > slow_ma:
            return "BUY"
        elif fast_ma < slow_ma:
            return "SELL"
        else:
            return "HOLD"
    
    def process_signal(self, signal, price, capital):
        """Process trading signal"""
        if self.paper_trading:
            return
        
        if not self.executor:
            self.executor = TradeExecutor(self.kraken, self.pair, self.kraken_pair)
    
    def get_win_rate(self):
        """Calculate win rate"""
        if len(self.trades) == 0:
            return 0.0
        wins = sum(1 for trade in self.trades if trade.get('pnl_usd', 0) > 0)
        return (wins / len(self.trades)) * 100
    
    def get_total_pnl(self):
        """Calculate total PnL"""
        return sum(trade.get('pnl_usd', 0) for trade in self.trades)
