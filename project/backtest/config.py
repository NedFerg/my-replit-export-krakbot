"""
Backtesting configuration and MODE switch.

Set the BOT_MODE environment variable to control execution:
    BOT_MODE=live      → uses Kraken API and executes real orders
    BOT_MODE=backtest  → uses historical data and simulates trades (default)
    BOT_MODE=sim       → existing multi-agent simulation mode

When MODE == "backtest":
    - All Kraken API calls are disabled
    - All real order execution is disabled
    - Price data is routed through HistoricalDataFeed
    - Trades are executed via PortfolioSimulator
    - All logging remains identical to live mode
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Primary mode switch
# ---------------------------------------------------------------------------
MODE: str = os.environ.get("BOT_MODE", "backtest")
# Valid values: "live", "backtest", "sim"

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
BACKTEST_ASSETS: list[str] = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "LINK/USD",
    "AVAX/USD",
    "HBAR/USD",
    "XLM/USD",
]

# Short-code form (without /USD) used for portfolio and file naming
ASSET_SYMBOLS: list[str] = [a.split("/")[0] for a in BACKTEST_ASSETS]

# ---------------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------------
BACKTEST_TIMEFRAMES: list[str] = ["1h", "4h", "1d"]
DEFAULT_TIMEFRAME: str = "1h"

# ---------------------------------------------------------------------------
# Date ranges to fetch / backtest
# ---------------------------------------------------------------------------
DATE_RANGES: list[tuple[str, str]] = [
    ("2019-01-01", "2021-12-31"),
    ("2024-01-01", "2025-12-31"),
]

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]          # repo root
DATA_DIR = ROOT_DIR / "data" / "historical"             # OHLCV CSVs
TEST_WINDOWS_DIR = ROOT_DIR / "data" / "test_windows"   # bull/bear windows
RESULTS_DIR = ROOT_DIR / "results" / "full_backtests"   # macro backtest output
WINDOW_TESTS_RESULTS_DIR = ROOT_DIR / "results" / "window_tests"  # micro test output

# Ensure directories exist at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)
(TEST_WINDOWS_DIR / "bull").mkdir(parents=True, exist_ok=True)
(TEST_WINDOWS_DIR / "bear").mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WINDOW_TESTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Portfolio / simulation parameters
# ---------------------------------------------------------------------------
INITIAL_USD: float = 10_000.0     # starting cash in USD
TAKER_FEE: float = 0.004          # Kraken taker fee: 0.40%
MAKER_FEE: float = 0.0016         # Kraken maker fee: 0.16%
SLIPPAGE: float = 0.001           # 0.10% simulated slippage per trade

# Maximum ETF exposure (proportion of total equity)
MAX_ETF_EXPOSURE: float = 0.30

# ---------------------------------------------------------------------------
# CCXT / Kraken data fetch parameters
# ---------------------------------------------------------------------------
EXCHANGE_ID: str = "kraken"
FETCH_LIMIT: int = 500            # candles per CCXT call (Kraken max)
FETCH_SLEEP_SECONDS: float = 1.0  # polite rate-limit pause between calls

# ---------------------------------------------------------------------------
# Window extraction parameters (for extract_test_windows.py)
# ---------------------------------------------------------------------------
WINDOW_HOURS: int = 6             # target window length in hours (4-8 range)
BULL_MIN_PCT: float = 3.0         # minimum % gain to qualify as bullish
BEAR_MIN_PCT: float = -3.0        # maximum % change to qualify as bearish
NUM_WINDOWS: int = 10             # number of bull and bear windows each

# ---------------------------------------------------------------------------
# Metrics parameters
# ---------------------------------------------------------------------------
RISK_FREE_RATE: float = 0.05      # annualised risk-free rate for Sharpe/Sortino
TRADING_HOURS_PER_YEAR: int = 8760  # crypto trades 24/7 (hours in a year)
