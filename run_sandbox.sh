#!/usr/bin/env bash
# =============================================================================
# run_sandbox.sh — Start KrakBot in full sandbox mode
# =============================================================================
#
# What this does:
#   • USE_BULL_BEAR_TRADER=true  — activates the 4-phase Bull/Bear strategy
#   • USE_PAPER_BROKER=true      — all fills are synthetic (no real orders)
#   • KRAKEN_SANDBOX=true        — orders would be validated only (belt+braces)
#   • BOT_MODE=live              — starts the live price-feed + trading loop
#
# Prices are fetched live from Kraken's public API (no API key needed).
# All trades, P&L, and positions are simulated locally in the paper account.
#
# Crypto spot trades 24/7.  ETP/ETF hedge positions (ETHU, ETHD, SLON, XXRP)
# are only opened when US markets are open (Mon-Fri 09:30-16:30 ET).
#
# Log files written to project/logs/:
#   paper_trades.csv      — every synthetic fill
#   trade_archive.db      — SQLite archive (trades + phase transitions + rotations)
#
# Keyboard commands (type in this terminal + Enter):
#   S  — print paper trading summary (trades, P&L, positions)
#   P  — print phase status (BTC confidence, signal scores, open positions)
#   Ctrl-C — graceful shutdown (prints final summary automatically)
#
# Scheduling options (set before running):
#
#   Start in N hours from now:
#       START_DELAY_HOURS=11 ./run_sandbox.sh
#
#   Auto-wait until next US ETP market open (Mon-Fri 09:30 ET):
#       WAIT_FOR_ETP_MARKET=true ./run_sandbox.sh
#
# Usage:
#   chmod +x run_sandbox.sh
#   ./run_sandbox.sh
# =============================================================================

set -e

# Change to repo root regardless of where the script is called from
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Scheduled start: sleep until the requested start time
# ---------------------------------------------------------------------------
# Option 1 — explicit delay: START_DELAY_HOURS=11 ./run_sandbox.sh
if [[ -n "${START_DELAY_HOURS:-}" ]]; then
    delay_sec=$(python3 -c "import math; print(math.ceil(float('${START_DELAY_HOURS}') * 3600))")
    start_at=$(date -d "+${delay_sec} seconds" "+%Y-%m-%d %H:%M:%S %Z" 2>/dev/null \
               || python3 -c "
import datetime, time
t = datetime.datetime.now() + datetime.timedelta(seconds=${delay_sec})
print(t.strftime('%Y-%m-%d %H:%M:%S local'))")
    echo "⏰  Scheduled start in ${START_DELAY_HOURS}h (${delay_sec}s)"
    echo "    Will start at: ${start_at}"
    echo "    Press Ctrl-C to cancel."
    echo ""
    # Print a countdown every 30 minutes so the terminal stays informative
    elapsed=0
    while (( elapsed < delay_sec )); do
        remaining=$(( delay_sec - elapsed ))
        remaining_h=$(( remaining / 3600 ))
        remaining_m=$(( (remaining % 3600) / 60 ))
        echo "    ⏳  ${remaining_h}h ${remaining_m}m remaining until start..."
        sleep_chunk=$(( remaining < 1800 ? remaining : 1800 ))
        sleep "${sleep_chunk}"
        elapsed=$(( elapsed + sleep_chunk ))
    done
    echo "⏰  Delay complete — starting KrakBot now."
    echo ""

# Option 2 — auto-wait for next ETP market open: WAIT_FOR_ETP_MARKET=true
elif [[ "${WAIT_FOR_ETP_MARKET:-false}" == "true" ]]; then
    delay_sec=$(python3 - <<'PYEOF'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math

ET = ZoneInfo("America/New_York")
now = datetime.now(ET)

# Find next weekday 09:30 ET
candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now >= candidate or now.weekday() >= 5:
    # Already past today's open (or weekend) — move to next weekday
    candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

seconds = max(0, (candidate - now).total_seconds())
print(math.ceil(seconds))
PYEOF
    )
    start_at=$(python3 -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York')
now = datetime.now(ET)
candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now >= candidate or now.weekday() >= 5:
    candidate += __import__('datetime').timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += __import__('datetime').timedelta(days=1)
print(candidate.strftime('%A %Y-%m-%d 09:30 ET'))")
    echo "⏰  WAIT_FOR_ETP_MARKET=true — sleeping until next US ETP market open"
    echo "    Next open: ${start_at}  (${delay_sec}s from now)"
    echo "    Press Ctrl-C to cancel."
    echo ""
    elapsed=0
    while (( elapsed < delay_sec )); do
        remaining=$(( delay_sec - elapsed ))
        remaining_h=$(( remaining / 3600 ))
        remaining_m=$(( (remaining % 3600) / 60 ))
        echo "    ⏳  ${remaining_h}h ${remaining_m}m until ETP market open..."
        sleep_chunk=$(( remaining < 1800 ? remaining : 1800 ))
        sleep "${sleep_chunk}"
        elapsed=$(( elapsed + sleep_chunk ))
    done
    echo "⏰  ETP market open — starting KrakBot now."
    echo ""
fi

echo "============================================================"
echo "  KrakBot — SANDBOX MODE"
echo "  Strategy : Bull/Bear Rotational Trader"
echo "  Broker   : PaperBroker (synthetic fills, zero real orders)"
echo "  Prices   : Live from Kraken public API"
echo "  Spot     : Crypto 24/7 (BTC, ETH, SOL, XRP, HBAR, LINK, XLM)"
echo "  ETPs     : Gated — only during US market hours (09:30-16:30 ET)"
echo "  Entry    : BTC rolling-high breakout from current price"
echo "             (floor=\$65K; works from any price above floor)"
echo "============================================================"
echo ""

USE_BULL_BEAR_TRADER=true \
USE_PAPER_BROKER=true \
KRAKEN_SANDBOX=true \
BOT_MODE=live \
BTC_BULL_RUN_FLOOR=65000 \
BREAKOUT_CONFIDENCE_MIN=0.55 \
python3 project/main.py
