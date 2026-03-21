"""
Backtesting framework for the Krakbot trading system.

Modules:
    config           - MODE switch and global constants
    data_loader      - CCXT-based historical data fetcher and CSV manager
    historical_feed  - HistoricalDataFeed: candle-by-candle data iterator
    portfolio_simulator - PortfolioSimulator: simulated order execution + P&L
    backtest_engine  - Main backtesting loop
    metrics          - Performance metrics (Sharpe, drawdown, win rate, etc.)
    plotter          - Equity curve and trade signal visualization
    runner           - Top-level backtest orchestrator
"""

from .config import MODE, BACKTEST_ASSETS, BACKTEST_TIMEFRAMES
from .historical_feed import HistoricalDataFeed
from .portfolio_simulator import PortfolioSimulator

__all__ = [
    "MODE",
    "BACKTEST_ASSETS",
    "BACKTEST_TIMEFRAMES",
    "HistoricalDataFeed",
    "PortfolioSimulator",
]
