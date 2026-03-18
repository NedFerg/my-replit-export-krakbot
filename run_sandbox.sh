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
#   • Runs 24/7 — crypto spot trades any time (BTC, ETH, SOL, XRP, HBAR, LINK, XLM)
#   • ETP/short-ETF trades (ETHD, SETH) fire only during US market hours (Mon-Fri 09:30-16:30 ET)
#   • The ETP gate is enforced in the strategy code automatically — no manual scheduling needed
#
# Log files written to project/logs/:
#   paper_trades.csv      — every synthetic fill (CSV, one row per trade)
#   trade_archive.db      — SQLite archive (trades + phase transitions + rotations)
#   eod_YYYYMMDD.txt      — end-of-day analysis report (written by Ctrl-C or AUTO_SHUTDOWN_ET)
#
# Keyboard commands (type in this terminal + Enter):
#   S  — print paper trading summary (trades, P&L, positions)
#   P  — print phase status (BTC confidence, signal scores, open positions)
#   Ctrl-C — graceful shutdown (saves EOD report and prints final summary)
#
# Options (set before running):
#
#   Stop automatically at 16:30 ET each day (ETP market close):
#       AUTO_SHUTDOWN_ET=true ./run_sandbox.sh
#
#   Wait until next weekday 09:30 ET before starting:
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
        "ETHD": "ETHDUSD",
        "SETH": "SETHUSD",
    }
    ALIASES = {
        "XXBTZUSD": "BTC",
        "XETHZUSD": "ETH",
        "XXRPZUSD": "XRP",
        "XXLMZUSD": "XLM",
    }

    # Sanity ranges — prices outside these mean something is wrong with the feed
    SANITY = {
        "BTC":  (30_000, 200_000),
        "ETH":  (500,    10_000),
        "SOL":  (10,     1_000),
        "XRP":  (0.10,   20),
        "LINK": (1,      100),
        "HBAR": (0.01,   1),
        "XLM":  (0.01,   5),
    }

    pairs_str = ",".join(PAIRS.values())
    url = "https://api.kraken.com/0/public/Ticker?" + urllib.parse.urlencode({"pair": pairs_str})

    print(f"    Fetching: {url}")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=8) as resp:
        raw = resp.read()
        data = json.loads(raw)
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
        print(f"    Raw response: {raw[:500]}")
        sys.exit(0)

    print()
    print("🔌  Live Kraken price feed — CONFIRMED  ✅")
    print(f"    Source  : api.kraken.com/0/public/Ticker  (last-trade 'c' field)")
    print(f"    Latency : {latency_ms} ms")
    print(f"    Assets  : {len(prices)}/{len(PAIRS)} responding")
    print()

    bad = []
    for asset in ["BTC", "ETH", "SOL", "XRP", "HBAR", "LINK", "XLM", "ETHD", "SETH"]:
        price = prices.get(asset)
        if price is None:
            print(f"    {asset:<6}  (not returned by Kraken)")
            continue
        lo, hi = SANITY.get(asset, (0, float("inf")))
        flag = "  ⚠️  OUT OF RANGE" if not (lo <= price <= hi) else ""
        print(f"    {asset:<6}  ${price:>12,.4f}{flag}")
        if flag:
            bad.append(f"{asset} ${price:,.2f} (expected ${lo:,.0f}–${hi:,.0f})")

    print()
    if bad:
        print("⚠️  PRICE SANITY FAILED — these prices look wrong:")
        for b in bad:
            print(f"      {b}")
        print("   Check your network connection and verify against your Kraken app.")
        print("   Do NOT start trading until prices match what you see on Kraken.")
    else:
        print("    ✅  All prices within expected ranges.")
        print("    All fills will be SIMULATED against these live prices.")
    print()

except Exception as e:
    print(f"⚠️   Price feed check failed: {e}")
    print("    The bot will retry automatically once the trading loop starts.")
    print()
PYEOF

# ---------------------------------------------------------------------------
# Optional: wait until next ETP market open (WAIT_FOR_ETP_MARKET=true)
# By default the bot starts immediately — crypto trades 24/7.
# ---------------------------------------------------------------------------
if [[ "${WAIT_FOR_ETP_MARKET:-false}" == "true" ]]; then
    delay_sec=$(python3 - <<'PYEOF'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math

ET = ZoneInfo("America/New_York")
now = datetime.now(ET)

# Find next weekday 09:30 ET
candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now >= candidate or now.weekday() >= 5:
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
    echo "    Press Ctrl-C to cancel, or Ctrl-C and rerun without WAIT_FOR_ETP_MARKET to start now."
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
  Prices   : Live from Kraken public API (crypto + ETHD/SETH ETPs)
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
  Spot trade: 24/7 — buys/sells alts based on signals any time
  Short ETFs: ETP market hours only (auto-gated in strategy)
  Shutdown  : runs until Ctrl-C  (AUTO_SHUTDOWN_ET=true to stop at 16:30)
  Ctrl-C    : Graceful shutdown (prints final P&L summary)
  Keys      : S = paper summary  |  P = phase + positions
  Review    : python3 review_performance.py  (any time)
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
