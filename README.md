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

## Dynamic Capital Scaling: From $0 to $50k+

> **Works with ANY starting capital amount — no minimums.**

The bot calculates every position size proportionally from whatever capital
you provide.  The same strategy code runs whether you start with $1 or $50 000.

### Position sizes at different capital levels

| Capital | `RISK_MAX_POSITION_SIZE_PCT` | Capital per Trade | Notes |
|--------:|-----------------------------:|------------------:|-------|
| $1      | 15%                          | ~$0.15            | Trades smallest available Kraken lot |
| $10     | 15%                          | ~$1.50            | |
| $50     | 15%                          | ~$7.50            | Default `TOTAL_TRADING_CAPITAL` |
| $100    | 15%                          | ~$15.00           | |
| $500    | 15%                          | ~$75.00           | |
| $1,000  | 15%                          | ~$150.00          | |
| $5,000  | 15%                          | ~$750.00          | |
| $10,000 | 15%                          | ~$1,500.00        | |
| $50,000 | 15%                          | ~$7,500.00        | |

Set your capital via the environment variable (no restart needed on DigitalOcean):

```bash
export TOTAL_TRADING_CAPITAL=500
```

### Risk parameter scaling

All risk limits scale automatically with `TOTAL_TRADING_CAPITAL`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `RISK_MAX_POSITION_SIZE_PCT` | 0.15 (15%) | Fraction of capital per trade |
| `RISK_MAX_NOTIONAL_USD` | auto (capital × 15%) | Hard USD cap per order |
| `RISK_MAX_DAILY_DRAWDOWN_PCT` | 0.10 (10%) | Pause new trades when daily loss hits 10% |
| `REINVESTMENT_PROFIT_THRESHOLD_PCT` | 0.25 (25%) | Reinvest profits after 25% growth milestone |

### Exponential growth loop

1. Bot starts with `TOTAL_TRADING_CAPITAL`
2. Each profitable trade adds to `total_capital`
3. When realised profits reach `REINVESTMENT_PROFIT_THRESHOLD_PCT` of the last baseline, `capital_per_trade` is automatically increased
4. The cycle repeats — compounding growth at every capital level

### Bearish signal logic (SELL / short ETF rotation)

The bot trades **both bull and bear** market setups:

- **BUY signal:** RSI < 30 + MA uptrend + MACD positive → buy spot (24/7)
- **SELL signal:** RSI > 70 + MA downtrend + MACD negative → sell spot,
  then optionally rotate into ETHD/SETH if market hours allow

### Short ETF trading (SETH / ETHD)

| Setting | Value |
|---------|-------|
| Assets | ETHD (1× short ETH), SETH (2× short ETH) on Kraken Spot |
| Market hours | **Mon–Fri, 9:30 AM – 4:30 PM EST only** |
| Weekend holding | Held over weekend; close on bull signal or drawdown hit only |
| Margin / futures | ❌ None — spot ETFs only (legal in US) |

**Toggle short ETF trading from the DigitalOcean console:**

```bash
# Disable SETH/ETHD (spot trading continues normally — 24/7 unaffected)
export ENABLE_SHORT_ETF_TRADING=false

# Re-enable
export ENABLE_SHORT_ETF_TRADING=true
```

When `ENABLE_SHORT_ETF_TRADING=false` the bot skips all SETH/ETHD logic
but continues trading spot crypto around the clock without interruption.

### Running the bot

```bash
# Paper trading (default — safe, no real orders)
python3 project_scripts/trading_bot_runner_v2.py

# Live trading with custom capital
TOTAL_TRADING_CAPITAL=500 ENABLE_LIVE_TRADING=true \
  python3 project_scripts/trading_bot_runner_v2.py

# Paper mode, short ETFs disabled
ENABLE_SHORT_ETF_TRADING=false \
  python3 project_scripts/trading_bot_runner_v2.py
```

See `.env.example` for the full list of environment variables with example
configurations for every capital level from $1 to $50 000+.
