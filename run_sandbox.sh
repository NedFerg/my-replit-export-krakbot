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
# Log files written to project/logs/:
#   paper_trades.csv      — every synthetic fill
#   trade_archive.db      — SQLite archive (trades + phase transitions + rotations)
#
# Keyboard commands (type in this terminal + Enter):
#   S  — print paper trading summary (trades, P&L, positions)
#   P  — print phase status (BTC confidence, signal scores, open positions)
#   Ctrl-C — graceful shutdown (prints final summary automatically)
#
# Usage:
#   chmod +x run_sandbox.sh
#   ./run_sandbox.sh
# =============================================================================

set -e

# Change to repo root regardless of where the script is called from
cd "$(dirname "$0")"

echo "============================================================"
echo "  KrakBot — SANDBOX MODE"
echo "  Strategy : Bull/Bear Rotational Trader"
echo "  Broker   : PaperBroker (synthetic fills, zero real orders)"
echo "  Prices   : Live from Kraken public API"
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
