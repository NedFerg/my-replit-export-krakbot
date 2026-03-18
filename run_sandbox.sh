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
# Capital: $1,000 paper money (set PAPER_CAPITAL env var to change)
#
# Prices are fetched live from Kraken's public API (no API key needed).
# All trades, P&L, and positions are simulated locally in the paper account.
#
# Trading schedule:
#   • By default the script waits until 09:30 ET (US market open) before
#     starting, then auto-shuts down at 16:30 ET (market close).
#   • An end-of-day report is saved automatically to project/logs/eod_YYYYMMDD.txt
#     so you can review tonight what the bot traded.
#
# Log files written to project/logs/:
#   paper_trades.csv      — every synthetic fill (CSV, one row per trade)
#   trade_archive.db      — SQLite archive (trades + phase transitions + rotations)
#   eod_YYYYMMDD.txt      — end-of-day analysis report (saved at 16:30 shutdown)
#
# Keyboard commands (type in this terminal + Enter):
#   S  — print paper trading summary (trades, P&L, positions)
#   P  — print phase status (BTC confidence, signal scores, open positions)
#   Ctrl-C — graceful shutdown (saves EOD report and prints final summary)
#
# Scheduling options (set before running):
#
#   Skip the 09:30 ET wait and start immediately:
#       SKIP_MARKET_WAIT=true ./run_sandbox.sh
#
#   Start in N hours from now instead:
#       START_DELAY_HOURS=2 ./run_sandbox.sh
#
#   Auto-wait until next weekday 09:30 ET (useful if running over the weekend):
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
# Live Kraken price feed connectivity check
# Fetches current prices from api.kraken.com right now and displays them
# so you can confirm the feed is live before the trading session begins.
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import sys
try:
    import urllib.request, urllib.parse, json, time

    PAIRS = {
        "BTC":  "XBTUSD",
        "ETH":  "ETHUSD",
        "SOL":  "SOLUSD",
        "XRP":  "XRPUSD",
        "HBAR": "HBARUSD",
        "LINK": "LINKUSD",
        "XLM":  "XLMUSD",
    }
    ALIASES = {
        "XXBTZUSD": "BTC",
        "XETHZUSD": "ETH",
        "XXRPZUSD": "XRP",
        "XXLMZUSD": "XLM",
    }

    pairs_str = ",".join(PAIRS.values())
    url = "https://api.kraken.com/0/public/Ticker?" + urllib.parse.urlencode({"pair": pairs_str})

    t0 = time.time()
    with urllib.request.urlopen(url, timeout=8) as resp:
        data = json.loads(resp.read())
    latency_ms = int((time.time() - t0) * 1000)

    if data.get("error"):
        print(f"⚠️   Kraken API error: {data['error']}")
        sys.exit(0)

    result = data.get("result", {})
    reverse = {v.upper(): k for k, v in PAIRS.items()}
    reverse.update(ALIASES)

    prices = {}
    for key, ticker in result.items():
        asset = reverse.get(key.upper())
        if asset:
            try:
                prices[asset] = float(ticker["c"][0])
            except Exception:
                pass

    if not prices:
        print("⚠️   Could not parse any prices from Kraken response.")
        sys.exit(0)

    print()
    print("🔌  Live Kraken price feed — CONFIRMED  ✅")
    print(f"    Source  : https://api.kraken.com/0/public/Ticker")
    print(f"    Latency : {latency_ms} ms")
    print(f"    Assets  : {len(prices)}/{len(PAIRS)} responding")
    print()
    for asset in ["BTC", "ETH", "SOL", "XRP", "HBAR", "LINK", "XLM"]:
        price = prices.get(asset)
        if price is not None:
            print(f"    {asset:<6}  ${price:>12,.4f}")
    print()
    print("    All fills will be SIMULATED against these live prices.")
    print("    No API key required — public feed only.")
    print()

except Exception as e:
    print(f"⚠️   Price feed check failed: {e}")
    print("    The bot will retry automatically once the trading loop starts.")
    print()
PYEOF

# ---------------------------------------------------------------------------
# Default: auto-wait until today's 9:30 AM ET market open
# Runs only when no other schedule flag is given AND the market hasn't opened.
# Override: set START_DELAY_HOURS or WAIT_FOR_ETP_MARKET=true to use those
# mechanisms instead; or set SKIP_MARKET_WAIT=true to start immediately.
# ---------------------------------------------------------------------------
if [[ -z "${START_DELAY_HOURS:-}" \
      && "${WAIT_FOR_ETP_MARKET:-false}" != "true" \
      && "${SKIP_MARKET_WAIT:-false}" != "true" ]]; then
    delay_sec=$(python3 - <<'PYEOF'
from datetime import datetime
from zoneinfo import ZoneInfo
import math

ET = ZoneInfo("America/New_York")
now = datetime.now(ET)

# Only wait if today is a weekday and we're before today's 9:30 AM open
target = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now.weekday() < 5 and now < target:
    print(math.ceil((target - now).total_seconds()))
else:
    print(0)
PYEOF
    )
    if (( delay_sec > 0 )); then
        start_at=$(python3 -c "
from datetime import datetime
from zoneinfo import ZoneInfo
ET = ZoneInfo('America/New_York')
now = datetime.now(ET)
print(now.replace(hour=9, minute=30, second=0, microsecond=0).strftime('%A %Y-%m-%d 09:30 ET'))")
        echo "⏰  Market opens at 09:30 ET — sleeping until then (${delay_sec}s)"
        echo "    Start time : ${start_at}"
        echo "    Crypto spot (BTC/ETH/SOL/…) will begin trading at open."
        echo "    Press Ctrl-C to cancel, or set SKIP_MARKET_WAIT=true to start now."
        echo ""
        elapsed=0
        while (( elapsed < delay_sec )); do
            remaining=$(( delay_sec - elapsed ))
            remaining_h=$(( remaining / 3600 ))
            remaining_m=$(( (remaining % 3600) / 60 ))
            echo "    ⏳  ${remaining_h}h ${remaining_m}m until 09:30 ET market open..."
            sleep_chunk=$(( remaining < 1800 ? remaining : 1800 ))
            sleep "${sleep_chunk}"
            elapsed=$(( elapsed + sleep_chunk ))
        done
        echo "⏰  09:30 ET — market open, starting KrakBot now."
        echo ""
    fi
fi

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

# ---------------------------------------------------------------------------
# Show current ETP market status — always displayed at launch
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
now = datetime.now(ET)
now_utc = datetime.now(timezone.utc)
weekday = now.weekday()
mkt_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
mkt_close = now.replace(hour=16, minute=30, second=0, microsecond=0)

# Bot clock — shown so the operator can confirm the server time is correct
clock_utc   = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
clock_et    = now.strftime("%Y-%m-%d %H:%M:%S ET")

# Paper capital — reads the same env var as config.py so they always agree
paper_capital = float(os.getenv("PAPER_CAPITAL", "1000"))
capital_str   = f"${paper_capital:,.2f}  (set PAPER_CAPITAL env var to change)"

if weekday < 5 and mkt_open <= now < mkt_close:
    etp_status = "OPEN  ✅  ETP/ETF trades are LIVE right now"
    until_close = int((mkt_close - now).total_seconds())
    ch, cm = divmod(until_close // 60, 60)
    etp_detail = f"Market closes in {ch}h {cm}m  (at 16:30 ET)"
elif weekday < 5 and now < mkt_open:
    wait = int((mkt_open - now).total_seconds())
    wh, wm = divmod(wait // 60, 60)
    etp_status = f"CLOSED — opens today in {wh}h {wm}m  (09:30 ET)"
    etp_detail = "Crypto spot is trading 24/7; ETP entries will unlock at open"
else:
    # Weekend or post-close
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    wait = int((candidate - now).total_seconds())
    wh, wm = divmod(wait // 60, 60)
    etp_status = f"CLOSED (weekend) — next open {candidate.strftime('%A')} in {wh}h {wm}m"
    etp_detail = f"Crypto spot is trading 24/7; ETP entries will unlock at {candidate.strftime('%A')} open"

print(f"""
============================================================
  KrakBot — SANDBOX MODE  (all fills are PAPER / simulated)
  Strategy : Bull/Bear Rotational Trader
  Broker   : PaperBroker — ZERO real orders sent to Kraken
  Prices   : Live from Kraken public API (no API key needed)
  ------------------------------------------------------------
  Bot clock: {clock_utc}
             {clock_et}
  Capital  : {capital_str}
  ------------------------------------------------------------
  Crypto   : 24/7 spot trading always active
             BTC, ETH, SOL, XRP, HBAR, LINK, XLM
  ETPs     : {etp_status}
             {etp_detail}
  Bear short: ETHD 2× inverse (15-25%)  +  SETH 1× inverse (5-8%)
  ------------------------------------------------------------
  Entry    : BTC rolling-high breakout (floor=$65K)
  Ctrl-C   : Graceful shutdown (prints final P&L summary)
  Keys     : S = paper summary  |  P = phase + positions
  Review   : python3 review_performance.py  (any time)
============================================================
""")
PYEOF

USE_BULL_BEAR_TRADER=true \
USE_PAPER_BROKER=true \
KRAKEN_SANDBOX=true \
BOT_MODE=live \
BTC_BULL_RUN_FLOOR=65000 \
BREAKOUT_CONFIDENCE_MIN=0.55 \
python3 project/main.py
