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
