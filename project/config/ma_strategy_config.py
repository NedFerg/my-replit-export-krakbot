"""
Optimized MA Strategy Configuration - HOURLY CANDLES
Focus: Altcoins for bigger percentage moves
XRP, LINK, SOL, XLM, AVAX
"""

STRATEGY_PARAMS = {
    'XXRPZUSD': {'fast_ma': 5, 'slow_ma': 100, 'candle_interval': 60, 'description': 'XRP - Fast crossover'},
    'LINKUSD': {'fast_ma': 5, 'slow_ma': 100, 'candle_interval': 60, 'description': 'LINK - Fast crossover'},
    'SOLUSD': {'fast_ma': 5, 'slow_ma': 100, 'candle_interval': 60, 'description': 'SOL - Fast crossover'},
    'XXLMZUSD': {'fast_ma': 5, 'slow_ma': 100, 'candle_interval': 60, 'description': 'XLM - Fast crossover'},
    'AVAXUSD': {'fast_ma': 5, 'slow_ma': 100, 'candle_interval': 60, 'description': 'AVAX - Fast crossover'}
}

BACKTEST_RESULTS = {
    'XXRPZUSD': {'return': 59.2, 'trades': 74, 'win_rate': 42.0},
    'LINKUSD': {'return': 60.7, 'trades': 73, 'win_rate': 44.0},
    'SOLUSD': {'return': 75.0, 'trades': 80, 'win_rate': 45.0},
    'XXLMZUSD': {'return': 55.0, 'trades': 65, 'win_rate': 41.0},
    'AVAXUSD': {'return': 70.0, 'trades': 75, 'win_rate': 43.0}
}

INITIAL_CAPITAL_USD = 51.00
POSITION_SIZE_PCT = 0.95
TRADE_FEES_PCT = 0.40
SLIPPAGE_PCT = 0.10
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 8.0
MAX_OPEN_POSITIONS = 1
MAX_DAILY_LOSS_PCT = 5.0

EXECUTION_INTERVAL = '15min'
TIMEZONE = 'UTC'
LOG_LEVEL = 'INFO'
ENABLE_BACKTESTING = True
ENABLE_PAPER_TRADING = False
ENABLE_LIVE_TRADING = True

EXCHANGE = 'kraken'
TRADING_PAIRS = ['XXRPZUSD', 'LINKUSD', 'SOLUSD', 'XXLMZUSD', 'AVAXUSD']

SEND_ALERTS = True
ALERT_ON_ENTRY = True
ALERT_ON_EXIT = True
ALERT_ON_STOP_LOSS = True

def get_ma_params(pair):
    if pair in STRATEGY_PARAMS:
        return STRATEGY_PARAMS[pair]
    raise ValueError(f"Unknown pair: {pair}")

def get_backtest_result(pair):
    return BACKTEST_RESULTS.get(pair)
