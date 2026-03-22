# Krakbot 🤖📈

Algorithmic crypto-trading bot for Kraken — trend-following long positions during the bull run, with short-ETF hedges (ETHD / SETH) for tactical downside protection.

---

## 📋 Project Manifest

> **→ [PROJECT_MANIFEST.md](./PROJECT_MANIFEST.md) ← Start here when resuming the project.**

The manifest is the single source of truth for:
- Project goals and trading philosophy
- Current status and known issues
- Architecture and design decisions
- Implementation roadmap (phases 1–5)
- Session-by-session progress log
- Quick reference commands and file locations
- Environment details and trading parameters

---

## Quick Start

```bash
# Paper trading (no API keys needed)
python3 project/main.py

# Watch live trades
tail -f project/logs/paper_trades.csv
```

---

## Supporting Docs

| File | Purpose |
|------|---------|
| [PROJECT_MANIFEST.md](./PROJECT_MANIFEST.md) | Master source of truth — read first every session |
| [PR_TRACKER.md](./PR_TRACKER.md) | Active PR status, blockers, merge order |
| [MEMENTO_SYSTEM.md](./MEMENTO_SYSTEM.md) | Protocol for keeping context docs up to date |
| [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) | Detailed architecture and trading thesis |
| [SESSION_MEMORY.md](./SESSION_MEMORY.md) | Key decisions and architectural insights |

---

## Scaling from $51 to $50k

The enhanced bot v2 (`project_scripts/trading_bot_runner_v2.py`) is designed to
work at **any account size** without changing any code.  Everything is controlled
through environment variables in `.env`.

### Step 1 — Copy the template

```bash
cp .env.example .env
```

### Step 2 — Set your capital and risk limits

| Account size | `TOTAL_TRADING_CAPITAL` | `RISK_MAX_NOTIONAL_USD` | `RISK_MAX_POSITION_SIZE_PCT` |
|---|---|---|---|
| $51 starter | `51` | `10` | `0.15` |
| $500 | `500` | `75` | `0.15` |
| $5 000 | `5000` | `750` | `0.15` |
| $50 000 | `50000` | `7500` | `0.15` |

The 15% rule keeps each position a safe slice of your total capital regardless
of account size.

### Risk parameter explanations

| Variable | Default | What it does |
|---|---|---|
| `RISK_MAX_POSITION_SIZE_PCT` | `0.15` | No single trade uses more than 15% of capital |
| `RISK_MAX_NOTIONAL_USD` | `100` | Absolute hard cap per order in USD |
| `RISK_MAX_DAILY_DRAWDOWN_PCT` | `0.10` | Bot stops opening new positions after 10% daily loss |
| `REINVESTMENT_PROFIT_THRESHOLD_PCT` | `0.25` | Auto-reinvest when profits reach 25% of base capital |

### Exponential growth strategy

1. Bot starts with `TOTAL_TRADING_CAPITAL` and trades fixed-size positions.
2. Each profitable SELL is recorded by `portfolio_manager.py`.
3. When **cumulative realized profit ≥ 25%** of base capital, the
   `capital_per_trade` is automatically increased to include a share of profits.
4. The next trade is slightly larger → produces slightly more profit → the cycle
   compounds over time.
5. The `RiskManager` always clamps the order to safe limits, so growth is
   controlled and never exceeds the per-position or daily-loss caps.

### Running the enhanced bot

```bash
# Paper mode (safe — no real orders)
TOTAL_TRADING_CAPITAL=500 python3 project_scripts/trading_bot_runner_v2.py

# Live mode (requires Kraken API key in .env)
ENABLE_LIVE_TRADING=true python3 project_scripts/trading_bot_runner_v2.py
```
