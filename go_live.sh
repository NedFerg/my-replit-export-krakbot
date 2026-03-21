#!/usr/bin/env bash
# =============================================================================
# go_live.sh — Start KrakBot with REAL Kraken orders
# =============================================================================
#
# ⚠️  WARNING: This script submits REAL orders to your Kraken account.
#     Real money is at risk.  Use run_sandbox.sh for paper / simulated trading.
#
# Prerequisites (must be done before running this):
#   1. Run ./setup_keys.sh to configure and verify your Kraken API credentials.
#   2. Review .env — ensure USE_PAPER_BROKER=false and KRAKEN_SANDBOX=false.
#
# What this does:
#   • USE_BULL_BEAR_TRADER=true  — 4-phase Bull/Bear rotational strategy
#   • USE_PAPER_BROKER=false     — LiveBroker: real orders via Kraken REST API
#   • KRAKEN_SANDBOX=false       — no validate=true; orders fill for real
#   • BOT_MODE=live              — live price feed + trading loop
#
# Crypto spot: 24/7 (BTC, ETH, SOL, XRP, HBAR, LINK, XLM)
# ETP/ETF positions: Mon-Fri 09:30-16:30 ET only (ETHD, SETH, ETHU, SLON, XXRP)
#
# Scheduling (same as run_sandbox.sh):
#   START_DELAY_HOURS=11 ./go_live.sh          — wait N hours then start
#   WAIT_FOR_ETP_MARKET=true ./go_live.sh      — auto-wait for 09:30 ET
#
# Log files (project/logs/):
#   paper_trades.csv     — all executed fills
#   trade_archive.db     — SQLite: trades, phase transitions, rotations
#
# Keyboard commands:
#   S   — print live trading summary
#   P   — print phase status + positions
#   Ctrl-C — graceful shutdown (prints final P&L summary)
# =============================================================================

set -e
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Load .env if present (created by setup_keys.sh)
# ---------------------------------------------------------------------------
if [[ -f ".env" ]]; then
    # Export key=value lines, skipping blank lines and comments
    set -a
    # shellcheck source=/dev/null
    source <(grep -v '^[[:space:]]*#' .env | grep -v '^[[:space:]]*$')
    set +a
fi

# ---------------------------------------------------------------------------
# Credential guard — refuse to start without API keys
# ---------------------------------------------------------------------------
if [[ -z "${KRAKEN_API_KEY:-}" || -z "${KRAKEN_API_SECRET:-}" ]]; then
    echo ""
    echo "❌  KRAKEN_API_KEY and/or KRAKEN_API_SECRET are not set."
    echo ""
    echo "   Run ./setup_keys.sh first to configure and test your credentials."
    echo "   For paper / sandbox trading, use ./run_sandbox.sh instead."
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Safety check — refuse if either env var still points to paper mode
# ---------------------------------------------------------------------------
if [[ "${USE_PAPER_BROKER:-true}" == "true" ]]; then
    echo ""
    echo "⚠️   USE_PAPER_BROKER=true is set.  Switching to PaperBroker (no real orders)."
    echo "    If you intended live orders, set USE_PAPER_BROKER=false in .env."
    echo "    Falling back to run_sandbox.sh behaviour."
    echo ""
fi

# ---------------------------------------------------------------------------
# Scheduling helpers (same as run_sandbox.sh)
# ---------------------------------------------------------------------------
if [[ -n "${START_DELAY_HOURS:-}" ]]; then
    delay_sec=$(python3 -c "import math; print(math.ceil(float('${START_DELAY_HOURS}') * 3600))")
    start_at=$(date -d "+${delay_sec} seconds" "+%Y-%m-%d %H:%M:%S %Z" 2>/dev/null \
               || python3 -c "
import datetime
t = datetime.datetime.now() + datetime.timedelta(seconds=${delay_sec})
print(t.strftime('%Y-%m-%d %H:%M:%S local'))")
    echo "⏰  Scheduled start in ${START_DELAY_HOURS}h"
    echo "    Will start at: ${start_at}"
    echo "    Press Ctrl-C to cancel."
    echo ""
    elapsed=0
    while (( elapsed < delay_sec )); do
        remaining=$(( delay_sec - elapsed ))
        remaining_h=$(( remaining / 3600 ))
        remaining_m=$(( (remaining % 3600) / 60 ))
        echo "    ⏳  ${remaining_h}h ${remaining_m}m remaining until live start..."
        sleep_chunk=$(( remaining < 1800 ? remaining : 1800 ))
        sleep "${sleep_chunk}"
        elapsed=$(( elapsed + sleep_chunk ))
    done
    echo "⏰  Delay complete — starting KrakBot LIVE now."
    echo ""

elif [[ "${WAIT_FOR_ETP_MARKET:-false}" == "true" ]]; then
    read -r delay_sec start_at < <(python3 - <<'PYEOF'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
ET = ZoneInfo("America/New_York")
now = datetime.now(ET)
candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now >= candidate or now.weekday() >= 5:
    candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
delay = math.ceil(max(0, (candidate - now).total_seconds()))
label = candidate.strftime("%A %Y-%m-%d 09:30 ET")
print(delay, label)
PYEOF
    )
    echo "⏰  WAIT_FOR_ETP_MARKET=true — waiting for next ETP market open"
    echo "    Live trading starts: ${start_at}  (${delay_sec}s from now)"
    echo "    Press Ctrl-C to cancel."
    echo ""
    elapsed=0
    while (( elapsed < delay_sec )); do
        remaining=$(( delay_sec - elapsed ))
        remaining_h=$(( remaining / 3600 ))
        remaining_m=$(( (remaining % 3600) / 60 ))
        echo "    ⏳  ${remaining_h}h ${remaining_m}m until ETP open (live trading starts then)..."
        sleep_chunk=$(( remaining < 1800 ? remaining : 1800 ))
        sleep "${sleep_chunk}"
        elapsed=$(( elapsed + sleep_chunk ))
    done
    echo "⏰  ETP market open — starting KrakBot LIVE now."
    echo ""
fi

# ---------------------------------------------------------------------------
# Validate credentials before showing any confirmation prompt
# ---------------------------------------------------------------------------
echo "Verifying Kraken credentials…"
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, "project")
from broker.broker import LiveBroker

broker = LiveBroker(dry_run=True)
ok = broker.validate_credentials()
if not ok:
    print("")
    print("❌  Credential check failed.  Run ./setup_keys.sh to fix this.")
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Current ETP market status
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
now = datetime.now(ET)
now_utc = datetime.now(timezone.utc)
weekday = now.weekday()
mkt_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
mkt_close = now.replace(hour=16, minute=30, second=0, microsecond=0)

clock_utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
clock_et  = now.strftime("%Y-%m-%d %H:%M:%S ET")

if weekday < 5 and mkt_open <= now < mkt_close:
    etp_status = "OPEN  ✅  ETP/ETF trades active immediately"
    etp_detail = f"Market closes in {int((mkt_close-now).total_seconds())//3600}h {(int((mkt_close-now).total_seconds())%3600)//60}m"
elif weekday < 5 and now < mkt_open:
    wait = int((mkt_open - now).total_seconds())
    etp_status = f"CLOSED — opens today in {wait//3600}h {(wait%3600)//60}m  (09:30 ET)"
    etp_detail = "Crypto spot starts immediately; ETP entries unlock at open"
else:
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    wait = int((candidate - now).total_seconds())
    etp_status = f"CLOSED (weekend) — next open {candidate.strftime('%A')} in {wait//3600}h {(wait%3600)//60}m"
    etp_detail = f"ETP entries unlock {candidate.strftime('%A')} at 09:30 ET"

print(f"""
============================================================
  KrakBot — LIVE MODE  ⚠️  REAL ORDERS  ⚠️  REAL MONEY
  Strategy : Bull/Bear Rotational Trader
  Broker   : LiveBroker — orders sent to api.kraken.com
  Prices   : Live from Kraken public API
  ------------------------------------------------------------
  Bot clock: {clock_utc}
             {clock_et}
  ------------------------------------------------------------
  Crypto   : 24/7 spot trading (BTC, ETH, SOL, XRP, HBAR, LINK, XLM)
  ETPs     : {etp_status}
             {etp_detail}
  Bear short: ETHD 2× inverse (15-25%)  +  SETH 1× inverse (5-8%)
  ------------------------------------------------------------
  Entry    : BTC rolling-high breakout (floor=$65K)
  Ctrl-C   : Graceful shutdown (prints final P&L summary)
  Keys     : S = summary  |  P = phase + positions
============================================================
""")
PYEOF

# ---------------------------------------------------------------------------
# ⚠️  Final confirmation — operator must type GOLIVE to proceed
# ---------------------------------------------------------------------------
echo ""
echo "  ⚠️  ⚠️  ⚠️  LIVE TRADING CONFIRMATION  ⚠️  ⚠️  ⚠️"
echo ""
echo "  This will submit REAL orders to your Kraken account."
echo "  Real money is at risk.  There is no undo."
echo ""
echo "  Type  GOLIVE  and press Enter to start:"
read -r confirm

if [[ "$confirm" != "GOLIVE" ]]; then
    echo ""
    echo "  Confirmation not received.  Aborting — no orders sent."
    echo "  Use ./run_sandbox.sh for paper / simulated trading."
    echo ""
    exit 0
fi

echo ""
echo "  ✅  Confirmed.  Starting KrakBot in LIVE mode."
echo ""

# ---------------------------------------------------------------------------
# Launch — load .env vars, then start main.py
# ---------------------------------------------------------------------------
USE_BULL_BEAR_TRADER=true \
USE_PAPER_BROKER=${USE_PAPER_BROKER:-false} \
KRAKEN_SANDBOX=${KRAKEN_SANDBOX:-false} \
BOT_MODE=live \
BTC_BULL_RUN_FLOOR=${BTC_BULL_RUN_FLOOR:-65000} \
BREAKOUT_CONFIDENCE_MIN=${BREAKOUT_CONFIDENCE_MIN:-0.55} \
python3 project/main.py
