#!/usr/bin/env python3
"""
Trading Bot Runner v2 - Enhanced Signal System
RSI + MACD + Support/Resistance
Manages existing positions + new bot trades

Environment variables consumed here:
  TOTAL_TRADING_CAPITAL            – starting capital in USD (default: 50)
  RISK_MAX_POSITION_SIZE_PCT       – max position as fraction of account (default: 0.15)
  RISK_MAX_NOTIONAL_USD            – max USD per position (default: 100)
  RISK_MAX_DAILY_DRAWDOWN_PCT      – max daily loss fraction (default: 0.10)
  REINVESTMENT_PROFIT_THRESHOLD_PCT– profit % that triggers reinvest (default: 0.25)
  ENABLE_LIVE_TRADING              – "true" / "false" override (default: from config)
"""

import logging
import os
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
from project_scripts.risk_manager import RiskManager
from project_scripts.portfolio_manager import PortfolioManager

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_total_capital(kraken_api) -> float:
    """
    Determine total trading capital in USD.

    Priority:
      1. TOTAL_TRADING_CAPITAL environment variable
      2. Live Kraken account balance (if API is enabled)
      3. Hard-coded fallback of $50
    """
    env_capital = os.environ.get("TOTAL_TRADING_CAPITAL")
    if env_capital:
        capital = float(env_capital)
        logger.info("Capital loaded from env TOTAL_TRADING_CAPITAL: $%.2f", capital)
        return capital

    if kraken_api.enabled:
        capital = kraken_api.get_account_balance() or 50.0
        logger.info("Capital loaded from Kraken balance: $%.2f", capital)
        return capital

    logger.info("Capital using default fallback: $50.00")
    return 50.0


class TradingBotRunnerV2:
    def __init__(self, paper_trading=True):
        self.paper_trading = paper_trading
        self.mode = "PAPER" if paper_trading else "LIVE"
        
        # Get real balances from Kraken
        kraken_api = KrakenAPI()
        self.kraken_api = kraken_api
        
        # ── Capital (env-first, then Kraken, then fallback) ──────────────────
        self.available_balance = _load_total_capital(kraken_api)
        
        # ── Portfolio manager (tracks PnL, drives reinvestment) ───────────────
        self.portfolio = PortfolioManager(base_capital=self.available_balance)

        # ── Risk manager (enforces limits before every order) ─────────────────
        self.risk = RiskManager(total_capital=self.available_balance)

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
            except Exception:
                pass
        return prices
    
    def _log_initialization(self):
        """Log portfolio initialization"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("[PORTFOLIO INITIALIZATION - ENHANCED BOT v2]")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Total Trading Capital: ${self.available_balance:.2f}")
        logger.info(f"Capital Per Trade:     ${self.portfolio.capital_per_trade:.2f}")
        logger.info("")
        logger.info("[RISK LIMITS]")
        rs = self.risk.status()
        logger.info(f"  Max position size: {rs['max_position_pct']*100:.0f}% of capital")
        logger.info(f"  Max notional:      ${rs['max_notional_usd']:.2f}")
        logger.info(f"  Max daily loss:    {rs['max_daily_drawdown_pct']*100:.0f}% of capital")
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
        logger.info(f"Trading Capital Available:     ${self.available_balance:.2f}")
        logger.info("=" * 80)
        logger.info("")
    
    def _compute_capital_per_trade(self) -> float:
        """
        Return the capital to deploy per position, sourced from the portfolio
        manager (which auto-scales with profits).  Clamped by risk manager.
        """
        raw = self.portfolio.capital_per_trade
        return self.risk.clamp_notional(raw)

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

                total_unrealized = 0.0
                total_realized = 0.0

                for asset_name, bot in sorted(self.bots.items()):
                    kraken_pair = bot.pair
                    
                    if kraken_pair not in feed_data:
                        continue

                    df = feed_data[kraken_pair]
                    if len(df) == 0:
                        continue

                    # Calculate signals
                    signal_data = bot.calculate_signals(df)
                    signal = signal_data.get('signal', 'HOLD')
                    reason = signal_data.get('reason', '')
                    confidence = signal_data.get('confidence', 0)

                    # Log signal
                    conf_str = f" [conf={confidence:.0f}]" if confidence else ""
                    logger.info(f"  [{asset_name:6s}] {signal:5s}{conf_str} | {reason}")

                    # ── Risk gate before BUY ─────────────────────────────────
                    if signal == "BUY":
                        capital_to_use = self._compute_capital_per_trade()
                        allowed, block_reason = self.risk.check_buy(
                            capital_to_use, asset=asset_name
                        )
                        if not allowed:
                            logger.warning(f"  [{asset_name:6s}] Trade skipped: {block_reason}")
                            continue
                        signal_data['_capital_override'] = capital_to_use

                    # Process signal; capture realized PnL from SELL orders
                    capital = signal_data.pop('_capital_override', self._compute_capital_per_trade())
                    executed, realized_pnl = bot.process_signal(signal_data, capital)

                    # ── Post-trade accounting ────────────────────────────────
                    if executed and signal == "SELL" and realized_pnl != 0:
                        self.portfolio.record_trade_profit(realized_pnl, asset=asset_name)
                        # Update risk manager's capital reference
                        if realized_pnl < 0:
                            self.risk.record_loss(abs(realized_pnl))
                        self.risk.update_capital(self.portfolio.total_equity())
                        # Sync bot's capital_per_trade with portfolio
                        for b in self.bots.values():
                            b.capital_per_trade = self.portfolio.capital_per_trade

                # ── Portfolio summary ─────────────────────────────────────────
                logger.info("")
                logger.info("=" * 80)
                logger.info("[PORTFOLIO STATUS]")
                logger.info("=" * 80)
                
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

                # Update portfolio manager with latest unrealized PnL
                self.portfolio.update_unrealized(total_unrealized)

                logger.info("")
                logger.info(f"Total Unrealized PnL: ${total_unrealized:+.2f}")
                logger.info(f"Total Realized PnL:   ${total_realized:+.2f}")
                logger.info(f"Available Cash:       ${self.available_balance:+.2f}")

                # Log full portfolio snapshot every iteration
                self.portfolio.log_snapshot()

                # Risk status
                rs = self.risk.status()
                logger.info(
                    "[RiskManager] Daily loss budget remaining: $%.2f",
                    rs['daily_loss_budget_remaining'],
                )
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
    # Allow env override of live-trading flag
    env_live = os.environ.get("ENABLE_LIVE_TRADING", "").lower()
    if env_live == "true":
        live_mode = True
    elif env_live == "false":
        live_mode = False
    else:
        live_mode = ENABLE_LIVE_TRADING

    logger.info("")
    logger.info("=" * 80)
    logger.info("[ENHANCED TRADING BOT v2] STARTED")
    logger.info(f"Mode: {'LIVE TRADING' if live_mode else 'PAPER TRADING'}")
    logger.info("Signals: RSI + MACD + Support/Resistance (BUY + SELL)")
    logger.info(f"Capital: ${os.environ.get('TOTAL_TRADING_CAPITAL', '(from Kraken/default)')}")
    logger.info("=" * 80)
    
    runner = TradingBotRunnerV2(paper_trading=not live_mode)
    runner.run()
