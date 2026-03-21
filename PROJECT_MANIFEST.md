# 📋 PROJECT MANIFEST — Krakbot

> **Read this first** when resuming the project. It is the single source of truth for goals, status, architecture, and session history.  
> **Update this** after every meaningful session or PR merge.

---

## Table of Contents

1. [Project Goals](#1-project-goals)
2. [Current Status](#2-current-status)
3. [Architecture & Design Decisions](#3-architecture--design-decisions)
4. [Implementation Roadmap](#4-implementation-roadmap)
5. [Session Summary Log](#5-session-summary-log)
6. [Quick Reference Links](#6-quick-reference-links)
7. [Environment & Parameters](#7-environment--parameters)

---

## 1. Project Goals

### Vision
Build an algorithmic crypto-trading bot that:
- **Runs 24/7 on Kraken** using live price feeds with no manual intervention
- **Captures the 9–12 month bull run** ahead of us by riding strong uptrends in BTC, ETH, SOL, and alt-coins
- **Hedges short-term tops** by buying inverse/short ETFs (ETHD, SETH) when overbought signals are clear, then selling those hedges when reversal signals confirm the next wave up
- **Is NOT a scalping bot** — 5–20 quality trades per week is the target; only enter when the setup is genuinely high-quality

### Trading Philosophy
- **Trend-following core:** enter on strength, hold through normal pullbacks, exit only when momentum deteriorates or overbought signals fire
- **Quality over frequency:** fewer, higher-conviction trades beat death-by-a-thousand-cuts from over-trading
- **Asymmetric upside:** position sizing favours staying long during bull conditions; hedges are tactical and temporary
- **Short-ETF hedges are defensive, not speculative:** ETHD / SETH positions are sized to protect profits during short-term tops, not to profit from a prolonged bear; exit them aggressively when oversold signals flash

### Scope (9–12 Month Horizon)
- Phase 1 → validate edge in paper trading (current)
- Phase 2 → go live on Kraken with small real-money positions
- Phase 3 → add ETHD / SETH hedging layer
- Phase 4 → full live production with risk controls

---

## 2. Current Status

### ✅ What's Working

| Component | Status | Notes |
|-----------|--------|-------|
| Paper-trading engine (`PaperBroker`) | ✅ Working | Synthetic fills with slippage + fees |
| Live price feed (Kraken public API) | ✅ Working | 8 pairs, ~1 s refresh |
| MA crossover signal generation | ✅ Working | 5-bar short / 20-bar long |
| Trade logging (CSV + SQLite) | ✅ Working | `project/logs/` |
| EOD session summary | ✅ Working | `project/logs/eod_YYYYMMDD.txt` |
| Kill switch (equity drawdown guard) | ✅ Working | Stops bot if equity drops too far |

### ⚠️ Known Issues

| Issue | Impact | Priority |
|-------|--------|----------|
| MA windows (5/20) too short → scalpy signals | Bot over-trades on noise | 🔴 High |
| No overbought / oversold indicators (RSI, Bollinger) | Can't detect short-term tops | 🔴 High |
| No multi-timeframe confirmation | Entries fire on 1-min noise | 🟡 Medium |
| ETHD / SETH hedging not yet implemented | Can't capture short-side tops | 🟡 Medium |
| No hold logic (exits too early on pullbacks) | Misses extended trend gains | 🟡 Medium |
| `pyproject.toml` bloat (~1000 PyTorch packages) | Slows CI, install fails | 🟡 Medium |
| `torch` import not lazy-loaded in `rl_agent.py` | Crash if PyTorch absent | 🟡 Medium |
| No real-money live integration tested | Unproven in production | 🔵 Low (next phase) |

### Paper Trading Results (Session 2026-03-21)

```
Starting equity : $1,000.00
Ending equity   : $998.85
Realized P&L    : $0.00   (no closed trades yet)
Unrealized P&L  : -$0.51
Fees paid       : $0.64
Trades filled   : 8 (all BUYs; no exits triggered)
```

All 8 positions are open long — exits not yet firing because MA windows are too long relative to tick frequency, and the market hasn't reversed enough bars to trigger a CROSS DOWN.

---

## 3. Architecture & Design Decisions

### Directory Map

```
my-replit-export-krakbot/
├── project/
│   ├── agents/
│   │   └── rl_agent.py         # MA signal engine + position sizing
│   ├── broker/
│   │   └── broker.py           # LiveBroker + PaperBroker (fills, fees, slippage)
│   ├── backtest/               # Historical simulation framework
│   │   ├── config.py           # BOT_MODE env var, paths
│   │   ├── backtest_engine.py
│   │   ├── historical_feed.py
│   │   ├── portfolio_simulator.py
│   │   ├── data_loader.py
│   │   ├── metrics.py
│   │   ├── plotter.py
│   │   └── runner.py
│   ├── logs/                   # Runtime output (gitignored)
│   │   ├── paper_trades.csv
│   │   ├── trade_archive.db
│   │   └── eod_YYYYMMDD.txt
│   └── main.py                 # Entry point / main trading loop
├── project_scripts/
│   ├── fetch_historical_data.py
│   ├── run_window_tests.py
│   └── run_full_backtests.py
├── results/
│   ├── full_backtests/         # metrics.json, plots, FULL_SUMMARY.txt
│   └── window_tests/           # traces, MICRO_SUMMARY.txt
├── PROJECT_MANIFEST.md         # ← this file
├── PROJECT_CONTEXT.md          # Detailed architecture notes
├── PR_TRACKER.md               # Active PR status
├── SESSION_MEMORY.md           # Key decisions & quotes
└── MEMENTO_SYSTEM.md           # Protocol for keeping docs fresh
```

### Key Design Decisions

| Decision | What | Why |
|----------|------|-----|
| PaperBroker first | Synthetic fills before live money | Validate edge without risk |
| Duck-typing for broker detection | `hasattr(broker, "paper_cash")` | Avoids tight coupling between agent and broker type |
| Public API only in paper mode | No Kraken credentials needed | Bot runs without any secrets configured |
| SQLite for trade archive | Simple, portable, zero-dependency | Analyse trades with standard SQL; SCP to local |
| MA crossover as baseline | Simplest momentum signal | Starting point before layering RSI, Bollinger, multi-TF |
| Short ETFs (ETHD/SETH) not futures | Futures disabled for US users | Regulatory constraint; ETFs available on Kraken |
| BOT_MODE env var | `live` / `backtest` / `sim` | Easy mode switching without code changes |
| 2% equity per position | Small initial sizing | Limits single-asset risk while paper trading |

### Planned Strategy Architecture (Next Phase)

```
Multi-Timeframe Signal Stack
├── Daily TF   → macro trend filter (only trade in direction of 20-day MA)
├── 4-hour TF  → entry confirmation (MA crossover on 4h bars)
├── 1-hour TF  → fine-tune entry timing (RSI not overbought)
└── Overbought detector → RSI(14) > 75 OR price > 2σ Bollinger upper
                        → triggers ETHD / SETH hedge buy
                        → exits hedge when RSI < 45 AND momentum reverses
```

---

## 4. Implementation Roadmap

### Phase 1 — Multi-Timeframe Foundation 🔄 In Progress
- [ ] Switch from tick-based to bar-based MA (1h / 4h candles)
- [ ] Fetch real OHLCV bars from Kraken REST API
- [ ] Implement 3-timeframe alignment before entry fires
- [ ] Add RSI(14) to filter overbought entries
- [ ] Add minimum hold-time to prevent premature exits

### Phase 2 — Strategy Rewrite
- [ ] Replace simple 5/20 MA with adaptive trend model
- [ ] Entry: short MA crosses long MA AND 4h trend confirms AND RSI not overbought
- [ ] Exit: short MA crosses down OR RSI > 75 (partial exit / hedge trigger)
- [ ] Add trailing stop (e.g., 8% below highest close since entry)
- [ ] Add position add-on logic (pyramid into winners)

### Phase 3 — Short-ETF Hedge Layer (ETHD / SETH)
- [ ] Detect short-term top signals: RSI(14) > 75, price > 2σ Bollinger upper, bearish divergence
- [ ] Buy ETHD (2× short ETH) or SETH (1× short ETH) on confirmed top signal
- [ ] Size hedge at 10–20% of total portfolio (not to exceed 30%)
- [ ] Exit hedge when RSI < 45 AND momentum shows reversal (bullish engulfing candle, MACD cross)
- [ ] Wire ETHD/SETH Kraken pair strings into `broker.py`

### Phase 4 — Live Testing
- [ ] Configure Kraken API keys (trade-only, no withdrawal permission)
- [ ] Run live bot with $50 real money per position (tiny sizing)
- [ ] Monitor live P&L vs paper P&L for 2+ weeks
- [ ] Gradually increase position sizes after 20+ live trades with positive expectancy
- [ ] Set hard max drawdown kill switch at 15% portfolio loss

### Phase 5 — Production Hardening
- [ ] Add alerting (Telegram or email) on trade fill, kill-switch trigger, daily summary
- [ ] Add Grafana / simple web dashboard for equity curve
- [ ] Automated daily backtest vs last 30 days to detect strategy degradation
- [ ] Auto-restart on crash (systemd or supervisor)

---

## 5. Session Summary Log

> Add a new entry after every meaningful session or PR merge.  
> Most recent at top.

---

### Session: 2026-03-21 (Paper Trading Validation)

**Goal:** Get paper trading running end-to-end with live prices.

**Accomplished:**
- [x] PR #22: Removed invalid ETP pairs (ETHU, SLON, XXRP, ETHD, SETH) from `broker.py` price feed — these don't exist on Kraken public ticker
- [x] PR #24: Fixed paper trading to skip `fetch_live_balances()` in paper mode; source `remaining_usd` from `broker.paper_cash`; detect paper mode via `hasattr(broker, "paper_cash")`
- [x] Bot ran successfully and placed 8 paper trades across SOL, ETH, XLM, AVAX, LINK, XRP, BTC, HBAR
- [x] Established strategy direction: trend-following, 5–20 quality trades/week, no scalping
- [x] Clarified ETHD/SETH use-case: buy on clear overbought signals, exit when oversold + reversal confirmed

**Key Decisions Made:**
- Bot should NOT be a scalping bot — 5-20 trades/week is acceptable
- Hold positions through normal pullbacks; only exit on genuine momentum failure
- ETHD/SETH are short-term tactical hedges, not directional bets; size at 10–20% max

**Blockers Encountered:**
- MA windows too short (5/20) → over-triggering on tick noise; need bar-based OHLCV
- No overbought filter → entries fire even at local tops

**Files Modified:**
- `project/broker/broker.py` (removed invalid pairs)
- `project/agents/rl_agent.py` (paper mode balance detection)

**Next Session Should Start With:**
1. Switch MA strategy to bar-based OHLCV (1h candles from Kraken REST)
2. Add RSI(14) overbought filter to suppress entries when RSI > 70
3. Backtest revised strategy against historical bull / bear windows

---

### Session: 2026-03-21 Earlier (PR Tracker & Memento System)

**Accomplished:**
- [x] Created Memento System (`MEMENTO_SYSTEM.md`, `PROJECT_CONTEXT.md`, `SESSION_MEMORY.md`, `PR_TRACKER.md`)
- [x] Documented PR #4 (foundation), PR #5 (sandbox hardening), PR #6 (ETF hedging), PR #10 (CI/backtest)
- [x] Established merge order: PR #4 → #5 → #6 → #10

**Blockers Encountered:**
- `pyproject.toml` contains ~1000 unnecessary PyTorch package entries — slows CI
- `torch` import in `rl_agent.py` not lazy-loaded — crashes if PyTorch absent

---

## 6. Quick Reference Links

### Key Files

| File | Purpose |
|------|---------|
| `project/main.py` | Main trading loop entry point |
| `project/broker/broker.py` | LiveBroker + PaperBroker; price feed; order fills |
| `project/agents/rl_agent.py` | MA crossover signal engine; position sizing |
| `project/backtest/config.py` | BOT_MODE, paths, window test results dir |
| `project_scripts/run_window_tests.py` | Run micro-level backtest windows |
| `project_scripts/run_full_backtests.py` | Run full historical backtests |

### Active PRs

| PR | Title | Status |
|----|-------|--------|
| #4 | 24/7 crypto spot trading + ETP short strategy | 🔴 Blocked (pyproject bloat, torch import) |
| #5 | Sandbox hardening (retry, kill switch, RL validation) | 🟡 Blocked (needs #4) |
| #6 | ETF hedging layer (ETHD/SETH) | 🟡 Blocked (needs #5) |
| #10 | CI backtest pipeline (synthetic OHLCV fallback) | 🟡 Blocked (Kraken firewall in CI) |

### Data Locations

| Data | Path |
|------|------|
| Paper trade CSV | `project/logs/paper_trades.csv` |
| Trade archive (SQLite) | `project/logs/trade_archive.db` |
| EOD session summary | `project/logs/eod_YYYYMMDD.txt` |
| Full backtest results | `results/full_backtests/` |
| Window test results | `results/window_tests/` |

### Useful Commands

```bash
# Start paper trading
python3 project/main.py

# View live trades
tail -f project/logs/paper_trades.csv

# Session summary
cat project/logs/eod_$(date +%Y%m%d).txt

# Asset P&L summary
sqlite3 project/logs/trade_archive.db \
  "SELECT asset, COUNT(*), ROUND(SUM(pnl),4), ROUND(SUM(fees),4)
   FROM trades GROUP BY asset ORDER BY SUM(pnl) DESC;"

# Download logs to local machine
scp -r root@<droplet-ip>:~/my-replit-export-krakbot/project/logs/ ./krakbot_logs/

# Run window backtests
python3 project_scripts/run_window_tests.py

# Run full backtests
python3 project_scripts/run_full_backtests.py
```

---

## 7. Environment & Parameters

### Deployment

| Setting | Value |
|---------|-------|
| Host | DigitalOcean Droplet (root@k-bot) |
| OS | Linux |
| Python | 3.x |
| Repo path | `~/my-replit-export-krakbot/` |
| Exchange | Kraken (paper mode: public API only; live mode: private API) |

### Current Trading Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting paper cash | $1,000 | Resets each run unless DB persists |
| Max position per asset | 2% of equity (~$20) | Conservative while paper trading |
| Max short-ETF allocation | 30% of portfolio | Hard cap; never exceed |
| Slippage | 0.05% | Applied to both entry and exit fills |
| Fee rate | ~0.04% per trade | Matches Kraken maker/taker |
| MA windows | 5-bar short / 20-bar long | ⚠️ Needs to move to bar-based OHLCV |
| Price refresh | ~1 second | Kraken public ticker |
| Traded pairs (paper) | BTC, ETH, SOL, XRP, LINK, HBAR, XLM, AVAX | 8 crypto pairs |
| Short-ETF pairs (planned) | ETHD, SETH | Not yet wired into paper mode |
| BOT_MODE | `live` / `backtest` / `sim` | Set via env var |

### Kraken Pair Strings

```python
# project/broker/broker.py  — kraken_pairs dict
"BTC":  "XXBTZUSD"
"ETH":  "XETHZUSD"
"SOL":  "SOLUSD"
"XRP":  "XXRPZUSD"
"LINK": "LINKUSD"
"HBAR": "HBARUSD"
"XLM":  "XXLMZUSD"
"AVAX": "AVAXUSD"
# ETHD / SETH — add when hedging layer implemented
```

---

*Last updated: 2026-03-21 · Updated by: Copilot*
