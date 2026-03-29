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

---

## 🐂 Bull Market Mode

**Bull Market Mode** is an optional configuration that relaxes risk caps and expands
positioning limits to maximise returns during a prolonged crypto bull market.

### How to enable

```bash
# Enable bull mode (takes effect on next bot start)
export BULL_MARKET_MODE=true

# Disable when the market regime changes (takes effect on next bot start)
export BULL_MARKET_MODE=false
```

The bot logs a startup warning whenever bull mode is active:

```
[BullBearTrader] ⚠️  Bull Market Mode enabled: risk and ETF allocation limits relaxed; see README.md for details.
```

### What changes

| Parameter | Standard | Bull mode | Notes |
|-----------|----------|-----------|-------|
| `MAX_ETF_ALLOCATION` | 30% | **50%** | ETF cap raised; set `MAX_ETF_ALLOCATION` env var to override |
| `LEVERAGE_ETF_MAX` (per ETF) | 20% | **35%** | Larger leveraged-ETF positions |
| `SPOT_ALT_MAX` — core alts¹ | 15% | **25%** | BTC, ETH, SOL, XRP receive higher cap |
| `SPOT_ALT_MAX` — other alts | 15% | 15% | HBAR, LINK, XLM, AVAX unchanged |
| `RISK_MAX_DAILY_DRAWDOWN_PCT` | 10% | **18%** | Tolerates larger intra-day swings |
| `REINVESTMENT_PROFIT_THRESHOLD_PCT` | 25% | **10%** | Compounds capital faster |
| RSI exit threshold | 80 | **90** | More patience before trimming winners |
| Trailing stop (≥50% gain) | — | **20% from local high** | Strong runners not prematurely stopped |
| Exit confirmation | topping | topping + **overbought_score ≥ 0.5** | MACD/S/R confirmation required |

¹ Core conviction alts are those with the deepest liquidity and clearest bull-cycle
macro correlation: **BTC, ETH, SOL, XRP**.

### What does NOT change

- All existing safety checks (kill-switch, API fallback guards, minimum notional checks)
- Paper trading mode / simulated fills
- The bear-market short-ETF layer (ETHD / SETH)
- The `ENABLE_SHORT_ETF_TRADING` flag
- Kraken canonical pair symbols

### Patient exit / rotation logic in bull mode

1. **Trailing stop** — For positions with an unrealised gain ≥ 50%, the bot will not
   exit until the current price drops ≥ 20% below the highest price recorded since
   entry.  A position that is still rising is never stopped out early.

2. **Momentum confirmation** — Even when the alt-pump detector fires a "topping" signal,
   the bot also checks whether the hedge overlay's `overbought_score` is ≥ 0.5.  This
   score combines RSI > 90, Bollinger Band upper-band breach, and resistance-level
   proximity — equivalent to requiring a MACD reversal or support-level loss before
   trimming.

### Fine-tuning

Individual env vars always override the bull-mode defaults:

```bash
# Override only the daily drawdown cap; keep all other bull-mode defaults
export BULL_MARKET_MODE=true
export RISK_MAX_DAILY_DRAWDOWN_PCT=0.15   # 15% instead of 18%
```

See `.env.example` for the full parameter reference.

---

## ⚠️ Kraken Trading Pairs — Use Exact Canonical Symbols

Kraken requires **exact canonical pair codes** for its API. Common short aliases
such as `BTCUSD`, `ETHUSD`, `XRPUSD`, or `XLMUSD` are **not accepted** and will
produce `[KRAKEN PUBLIC ERROR] ['EQuery:Unknown asset pair']` errors in live mode.

**Always use the official Kraken symbols listed below:**

| Coin | Correct Kraken Symbol | ❌ Do NOT use |
|------|----------------------|--------------|
| BTC/USD | `XXBTZUSD` | BTCUSD, XBTUSD |
| ETH/USD | `XETHZUSD` | ETHUSD |
| SOL/USD | `SOLUSD`   | — |
| XRP/USD | `XXRPZUSD` | XRPUSD |
| XLM/USD | `XXLMZUSD` | XLMUSD |
| AVAX/USD | `AVAXUSD` | — |
| HBAR/USD | `HBARUSD` | — |
| LINK/USD | `LINKUSD`  | — |

In `.env` (or when overriding via the shell):

```bash
TRADING_PAIRS=XXBTZUSD,XETHZUSD,SOLUSD,XXRPZUSD,XXLMZUSD,AVAXUSD,HBARUSD,LINKUSD
```

Copying `.env.example` already sets this correctly. If you have an existing `.env`
from before this fix, update the `TRADING_PAIRS` line to match the list above.

