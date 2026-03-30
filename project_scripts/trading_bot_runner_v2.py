#!/usr/bin/env python3
"""
Trading Bot Runner v2 - Enhanced Signal System
RSI + MACD + Support/Resistance
Manages existing positions + new bot trades

Features
--------
* Dynamic capital scaling — works at ANY amount (no minimums)
* Risk manager enforces position size / drawdown limits before every order
* Portfolio manager tracks growth and drives profit reinvestment
* Full bearish signal logic: SELL spot + optional SETH/ETHD rotation
* SETH/ETHD guarded behind market-hours check AND ENABLE_SHORT_ETF_TRADING flag
* Spot ETH trading continues 24/7 regardless of ETF toggle state
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
from project.utils.market_hours import MarketHours, MarketSession
from project_scripts.trading_bot_live import KrakenAPI
from project_scripts.trading_bot_live_v2 import EnhancedTradeBot
from project_scripts.portfolio_manager import PortfolioManager
from project.broker.etf_hedging import ALL_ETFS, LONG_ETFS, SHORT_ETFS

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration — NO hardcoded minimums
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "true" if default else "false").lower()
    return val not in ("false", "0", "no")


# Capital the bot is allowed to deploy. Works at $1, $10, $100, $10k, $50k …
TOTAL_TRADING_CAPITAL: float = _env_float("TOTAL_TRADING_CAPITAL", 50.0)

# Risk parameters
RISK_MAX_POSITION_SIZE_PCT: float = _env_float("RISK_MAX_POSITION_SIZE_PCT", 0.15)
RISK_MAX_DAILY_DRAWDOWN_PCT: float = _env_float("RISK_MAX_DAILY_DRAWDOWN_PCT", 0.10)
REINVESTMENT_PROFIT_THRESHOLD_PCT: float = _env_float(
    "REINVESTMENT_PROFIT_THRESHOLD_PCT", 0.25
)

# Short ETF trading toggle (default ON; disable via env to keep spot-only mode)
ENABLE_SHORT_ETF_TRADING: bool = _env_bool("ENABLE_SHORT_ETF_TRADING", True)

# Use live trading if requested via env (overrides config file)
_LIVE_TRADING: bool = _env_bool("ENABLE_LIVE_TRADING", ENABLE_LIVE_TRADING)

_market_hours = MarketHours()


class LiveRiskManager:
    """
    Lightweight risk gate for the live trading runner.

    Checks each proposed trade against:
    1. Max position size (% of current portfolio capital)
    2. Max notional per trade (absolute USD cap)
    3. Daily drawdown limit (delegated to PortfolioManager)

    All limits scale proportionally with capital — no hardcoded thresholds.
    """

    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio
        self.violations: list[dict] = []

    def approve(self, asset: str, proposed_usd: float) -> tuple[bool, str]:
        """
        Return (approved, reason).

        Parameters
        ----------
        asset        : human-readable asset name for logging
        proposed_usd : USD notional of the proposed trade
        """
        allowed, reason = self.portfolio.check_trade_allowed(proposed_usd)
        if not allowed:
            self.violations.append(
                {
                    "asset": asset,
                    "proposed_usd": proposed_usd,
                    "reason": reason,
                    "time": datetime.utcnow().isoformat(),
                }
            )
            logger.warning(
                "[RiskManager] BLOCKED %s $%.2f — %s",
                asset, proposed_usd, reason,
            )
        return allowed, reason

    def log_summary(self) -> None:
        if self.violations:
            logger.info(
                "[RiskManager] Total violations this session: %d", len(self.violations)
            )


class TradingBotRunnerV2:
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.mode = "PAPER" if paper_trading else "LIVE"

        # ----------------------------------------------------------------
        # Kraken API + balance
        # ----------------------------------------------------------------
        kraken_api = KrakenAPI()
        self.kraken_api = kraken_api

        # Use TOTAL_TRADING_CAPITAL from env; fall back to live balance query
        if kraken_api.enabled:
            live_balance = kraken_api.get_account_balance() or TOTAL_TRADING_CAPITAL
        else:
            live_balance = TOTAL_TRADING_CAPITAL

        # TOTAL_TRADING_CAPITAL env var wins if explicitly set; otherwise use
        # live balance so the bot automatically adjusts to account growth.
        env_capital_set = "TOTAL_TRADING_CAPITAL" in os.environ
        self.available_balance: float = (
            TOTAL_TRADING_CAPITAL if env_capital_set else live_balance
        )

        self.all_balances: dict = {}
        if kraken_api.enabled:
            self.all_balances = kraken_api.get_all_balances()

        # ----------------------------------------------------------------
        # Portfolio manager (scales position sizes from any capital level)
        # ----------------------------------------------------------------
        self.portfolio = PortfolioManager(
            total_capital=self.available_balance,
            risk_pct=RISK_MAX_POSITION_SIZE_PCT,
            max_daily_drawdown_pct=RISK_MAX_DAILY_DRAWDOWN_PCT,
            reinvestment_threshold_pct=REINVESTMENT_PROFIT_THRESHOLD_PCT,
        )

        # ----------------------------------------------------------------
        # Risk manager
        # ----------------------------------------------------------------
        self.risk_manager = LiveRiskManager(self.portfolio)

        # ----------------------------------------------------------------
        # Asset map
        # ----------------------------------------------------------------
        self.asset_map = {
            "XXBT":  {"name": "BTC",  "pair": "XXBTZUSD"},
            "XETH":  {"name": "ETH",  "pair": "XETHZUSD"},
            "XXRP":  {"name": "XRP",  "pair": "XXRPZUSD"},
            "XXLM":  {"name": "XLM",  "pair": "XXLMZUSD"},
            "SOL":   {"name": "SOL",  "pair": "SOLUSD"},
            "AVAX":  {"name": "AVAX", "pair": "AVAXUSD"},
            "HBAR":  {"name": "HBAR", "pair": "HBARUSD"},
            "LINK":  {"name": "LINK", "pair": "LINKUSD"},
        }

        self.current_prices: dict = self._fetch_all_prices()
        self.feed = KrakenLiveFeed(pairs=TRADING_PAIRS)

        # ----------------------------------------------------------------
        # Initialise per-asset bots
        # ----------------------------------------------------------------
        self.bots: dict[str, EnhancedTradeBot] = {}
        for kraken_code, info in self.asset_map.items():
            kraken_pair = info["pair"]
            asset_name = info["name"]

            if kraken_pair not in TRADING_PAIRS:
                continue

            existing_amount = float(self.all_balances.get(kraken_code, 0))
            current_price = self.current_prices.get(kraken_pair, 0)

            try:
                bot = EnhancedTradeBot(
                    asset_name=asset_name,
                    kraken_pair=kraken_pair,
                    existing_amount=existing_amount,
                    current_price=current_price,
                    paper_trading=paper_trading,
                )
                self.bots[asset_name] = bot
            except Exception as exc:
                logger.error("Failed to create bot for %s: %s", asset_name, exc)

        self._log_initialization()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_all_prices(self) -> dict:
        """Fetch current prices for all tracked assets."""
        prices: dict = {}
        for kraken_code, info in self.asset_map.items():
            pair = info["pair"]
            try:
                resp = requests.get(
                    "https://api.kraken.com/0/public/Ticker",
                    params={"pair": pair},
                    timeout=5,
                )
                result = resp.json()
                if result.get("result"):
                    ticker = list(result["result"].values())[0]
                    prices[pair] = float(ticker["c"][0])
            except Exception:
                pass
        return prices

    def _etf_short_allowed(self) -> bool:
        """Return True if SETH/ETHD orders are allowed right now."""
        if not ENABLE_SHORT_ETF_TRADING:
            return False
        return _market_hours.get_session() == MarketSession.REGULAR

    def _log_initialization(self) -> None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("[PORTFOLIO INITIALIZATION - ENHANCED BOT v2]")
        logger.info("=" * 80)
        logger.info("Mode: %s", self.mode)
        logger.info("Total Capital (USD): $%.2f", self.available_balance)
        logger.info("Capital per Trade:   $%.2f", self.portfolio.capital_per_trade())
        logger.info(
            "Short ETF Trading:   %s",
            "ENABLED (SETH/ETHD, M-F 9:30-16:00 ET)"
            if ENABLE_SHORT_ETF_TRADING
            else "DISABLED (spot-only mode)",
        )
        logger.info("")
        logger.info("[TRADING BOTS] (with position tracking)")

        total_existing_value = 0.0
        for asset_name, bot in sorted(self.bots.items()):
            summary = bot.get_summary()
            if summary:
                logger.info(
                    "  %s: %.8f @ $%.4f",
                    asset_name,
                    summary["total_size"],
                    summary["avg_entry_price"],
                )
                total_existing_value += summary["unrealized_pnl"] + (
                    summary["avg_entry_price"] * summary["total_size"]
                )

        logger.info("")
        logger.info("Existing Holdings Value: $%.2f", total_existing_value)
        logger.info("=" * 80)
        logger.info("")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        iteration = 0

        while True:
            iteration += 1

            try:
                logger.info("")
                logger.info("=" * 80)
                logger.info(
                    ">>> ITERATION %d — %s UTC",
                    iteration,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                )
                logger.info("Market hours: %s", _market_hours.status_line())
                logger.info("=" * 80)

                # Check daily drawdown before processing any signals
                if self.portfolio.daily_drawdown_breached():
                    logger.warning(
                        "Daily drawdown limit reached — skipping signal processing."
                    )
                    time.sleep(300)
                    continue

                feed_data = self.feed.fetch_all_pairs(interval=60)
                capital_per_trade = self.portfolio.capital_per_trade()
                etf_short_now = self._etf_short_allowed()
                mkt_line = _market_hours.status_line()

                # ETF attempt tracking — reset each cycle, populated below
                # Keys: ETF ticker from ALL_ETFS; values: (status_tag, reason_str)
                _etf_cycle_status: dict[str, tuple[str, str]] = {}

                logger.info("")
                logger.info("[SIGNAL ANALYSIS] capital_per_trade=$%.2f", capital_per_trade)

                total_unrealized = 0.0
                total_realized = 0.0

                for asset_name, bot in sorted(self.bots.items()):
                    kraken_pair = bot.pair

                    if kraken_pair not in feed_data:
                        continue
                    df = feed_data[kraken_pair]
                    if len(df) == 0:
                        continue

                    # ---- Calculate signals ----
                    signal_data = bot.calculate_signals(df)
                    signal = signal_data.get("signal", "HOLD")
                    reason = signal_data.get("reason", "")
                    confidence = signal_data.get("confidence", 0.0)
                    is_bearish = signal_data.get("is_bearish", False)

                    logger.info(
                        "  [%s] %s (conf=%.2f) | %s",
                        asset_name, signal, confidence, reason,
                    )

                    # ---- Risk check ----
                    if signal in ("BUY", "SELL"):
                        approved, risk_reason = self.risk_manager.approve(
                            asset_name, capital_per_trade
                        )
                        if not approved:
                            logger.info(
                                "  [%s] Trade blocked by risk manager: %s",
                                asset_name, risk_reason,
                            )
                            continue

                    # ---- Execute trade ----
                    if signal == "BUY":
                        traded = bot.process_signal(signal_data, capital_per_trade)
                        if traded:
                            self.portfolio.on_trade_open(capital_per_trade)

                    elif signal == "SELL":
                        # Always sell spot position if one exists
                        traded = bot.process_signal(signal_data, capital_per_trade)
                        if traded:
                            # Retrieve PnL from the last closed trade
                            summary = bot.get_summary()
                            realized = summary.get("realized_pnl", 0.0) if summary else 0.0
                            self.portfolio.on_trade_close(realized)

                        # Optionally rotate into SETH/ETHD if market hours allow
                        if is_bearish and etf_short_now:
                            self._rotate_to_etf_short(asset_name, capital_per_trade)
                            # Record short-ETF attempt for the status report
                            for _etf in ("ETHD", "SETH"):
                                _prev = _etf_cycle_status.get(_etf)
                                _entry = (
                                    "ATTEMPTED",
                                    f"bear signal from {asset_name}"
                                    f" — ETF short rotation triggered",
                                )
                                if _prev is None or _prev[0] != "ATTEMPTED":
                                    _etf_cycle_status[_etf] = _entry
                        elif is_bearish and ENABLE_SHORT_ETF_TRADING and not etf_short_now:
                            logger.info(
                                "  [%s] Bear signal — ETF rotation deferred (outside market hours)",
                                asset_name,
                            )
                            for _etf in ("ETHD", "SETH"):
                                if _etf not in _etf_cycle_status:
                                    _etf_cycle_status[_etf] = (
                                        "DEFERRED",
                                        f"bear signal from {asset_name}"
                                        f" — outside ETF market hours ({mkt_line})",
                                    )

                    # Collect unrealized PnL
                    summary = bot.get_summary()
                    if summary:
                        total_unrealized += summary.get("unrealized_pnl", 0.0)
                        total_realized += summary.get("realized_pnl", 0.0)

                # Update portfolio unrealized PnL
                self.portfolio.update_unrealized(total_unrealized)

                # ---- Portfolio summary ----
                logger.info("")
                logger.info("=" * 80)
                logger.info("[PORTFOLIO STATUS]")
                logger.info("=" * 80)

                for asset_name, bot in sorted(self.bots.items()):
                    summary = bot.get_summary()
                    if summary:
                        logger.info("  %s:", asset_name)
                        logger.info("    Position:       %.8f", summary["total_size"])
                        logger.info("    Entry:          $%.4f", summary["avg_entry_price"])
                        logger.info("    Current:        $%.4f", summary["current_price"])
                        logger.info(
                            "    Unrealized PnL: $%+.2f (%+.2f%%)",
                            summary["unrealized_pnl"],
                            summary["unrealized_pnl_pct"],
                        )
                        logger.info("    Realized PnL:   $%+.2f", summary["realized_pnl"])

                logger.info("")
                logger.info("Total Unrealized PnL: $%+.2f", total_unrealized)
                logger.info("Total Realized PnL:   $%+.2f", total_realized)
                logger.info("Available Cash:       $%.2f", self.available_balance)
                self.portfolio.log_summary()
                self.risk_manager.log_summary()
                logger.info("=" * 80)

                # ---- ETF STATUS REPORT ----
                # Shows what the ETF overlay logic attempted or decided for
                # every ETF in ALL_ETFS this cycle.
                logger.info("")
                logger.info("=" * 80)
                logger.info("[ETF STATUS REPORT]")
                logger.info("=" * 80)
                logger.info("  Market  : %s", mkt_line)
                _etf_short_status = (
                    "ENABLED"
                    if ENABLE_SHORT_ETF_TRADING
                    else "DISABLED (set ENABLE_SHORT_ETF_TRADING=true to activate)"
                )
                logger.info("  Short ETF trading : %s", _etf_short_status)
                logger.info("")

                for _etf in ALL_ETFS:
                    if _etf in _etf_cycle_status:
                        _tag, _why = _etf_cycle_status[_etf]
                    elif _etf in LONG_ETFS:
                        _tag = "SKIPPED"
                        _why = (
                            "long ETF allocation not triggered this cycle"
                            " — no bullish spot signal produced an ETF entry"
                            " (long ETF rotation uses the RL/overlay path)"
                        )
                    elif _etf in SHORT_ETFS and not ENABLE_SHORT_ETF_TRADING:
                        _tag = "DISABLED"
                        _why = "ENABLE_SHORT_ETF_TRADING=false — short ETF trading is turned off"
                    elif _etf in SHORT_ETFS and not etf_short_now:
                        _tag = "DEFERRED"
                        _why = f"no bear signal triggered ETF rotation this cycle | market: {mkt_line}"
                    else:
                        _tag = "NO ACTION"
                        _why = "no bearish signal triggered an ETF short rotation this cycle"

                    logger.info("  %-6s: [%-8s] %s", _etf, _tag, _why)

                logger.info("=" * 80)

                # Wait for next cycle
                interval_seconds = 300
                logger.info("Next update in %ds…", interval_seconds)
                logger.info("")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("")
                logger.info("Bot interrupted by user")
                break
            except Exception as exc:
                logger.error("Error in main loop: %s", exc, exc_info=True)
                time.sleep(60)

    # ------------------------------------------------------------------
    # ETF rotation helper
    # ------------------------------------------------------------------

    def _rotate_to_etf_short(self, asset_name: str, usd_amount: float) -> None:
        """
        Placeholder for SETH/ETHD rotation logic.

        In production this would place a market buy order on Kraken for
        ETHD (1× short) or SETH (2× short) using `usd_amount` USD.
        The exact ETF chosen can be configurable.

        This method is guarded by both ENABLE_SHORT_ETF_TRADING and the
        market-hours check before it is ever called.
        """
        logger.info(
            "  [%s] ETF rotation → ETHD/SETH | amount=$%.2f "
            "(PAPER — no real order placed)",
            asset_name,
            usd_amount,
        )
        # TODO: integrate Kraken API call for ETHD/SETH here when live


if __name__ == "__main__":
    logger.info("")
    logger.info("=" * 80)
    logger.info("[ENHANCED TRADING BOT v2] STARTED")
    logger.info("Mode:    %s", "LIVE TRADING" if _LIVE_TRADING else "PAPER TRADING")
    logger.info("Capital: $%.2f", TOTAL_TRADING_CAPITAL)
    logger.info("Signals: RSI + MACD + Support/Resistance")
    logger.info("=" * 80)

    runner = TradingBotRunnerV2(paper_trading=not _LIVE_TRADING)
    runner.run()

