#!/usr/bin/env python3
"""
KrakBot — Session Performance Reviewer
=======================================
Run this any time to see how the current (or most recent) sandbox session
performed.  Reads:

  project/logs/paper_trades.csv   — PaperBroker fills (current session)
  project/trades.jsonl            — RL-agent trade journal (older sessions)
  project/logs/eod_YYYYMMDD.txt  — End-of-day report if the bot has closed

Usage:
  python3 review_performance.py             # auto-finds logs
  python3 review_performance.py --all       # include all historical JSONL trades
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(ROOT, "project", "logs", "paper_trades.csv")
JSONL_PATH = os.path.join(ROOT, "project", "trades.jsonl")
LOGS_DIR   = os.path.join(ROOT, "project", "logs")

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_usd(v: float, signed: bool = False) -> str:
    prefix = "+" if (signed and v >= 0) else ""
    return f"{prefix}${v:,.4f}"


def _fmt_pct(v: float, signed: bool = True) -> str:
    prefix = "+" if (signed and v >= 0) else ""
    return f"{prefix}{v:.2f}%"


def _parse_utc(ts_str: str) -> datetime:
    """Parse ISO-8601 UTC timestamp from the CSV."""
    ts_str = ts_str.rstrip("Z")
    return datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Read PaperBroker CSV (current session)
# ---------------------------------------------------------------------------

def load_paper_trades(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({
                    "timestamp":       _parse_utc(row["timestamp"]),
                    "asset":           row["asset"],
                    "side":            row["side"],
                    "size_coins":      float(row["size_coins"]),
                    "fill_price":      float(row["fill_price"]),
                    "notional_usd":    float(row["notional_usd"]),
                    "fee_usd":         float(row["fee_usd"]),
                    "realized_pnl_usd": float(row["realized_pnl_usd"]),
                    "position_after":  float(row["position_after_trade"]),
                    "source":          "paper",
                })
            except (KeyError, ValueError):
                pass
    return rows


# ---------------------------------------------------------------------------
# Read RL-agent JSONL (older / alternative execution path)
# ---------------------------------------------------------------------------

def load_jsonl_trades(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts  = datetime.fromtimestamp(rec.get("timestamp", 0), tz=timezone.utc)
                has_txid = bool(
                    rec.get("result") and rec["result"].get("txid")
                )
                rows.append({
                    "timestamp":    ts,
                    "asset":        rec.get("symbol", "?"),
                    "side":         rec.get("side", "?"),
                    "size_coins":   float(rec.get("size", 0)),
                    "fill_price":   0.0,          # not stored in JSONL
                    "notional_usd": 0.0,
                    "fee_usd":      0.0,
                    "realized_pnl_usd": 0.0,
                    "position_after": 0.0,
                    "source":       "live" if has_txid else "rl_agent",
                    "txid":         (rec.get("result") or {}).get("txid", []),
                    "kraken_descr": (rec.get("result") or {}).get(
                        "descr", {}).get("order", ""),
                })
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise_paper_trades(trades: list[dict]) -> None:
    if not trades:
        print("  No PaperBroker trades found in paper_trades.csv.")
        print()
        return

    first_ts = min(t["timestamp"] for t in trades)
    last_ts  = max(t["timestamp"] for t in trades)
    first_et = first_ts.astimezone(ET)
    last_et  = last_ts.astimezone(ET)

    buys  = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]

    total_notional = sum(t["notional_usd"] for t in trades)
    total_fees     = sum(t["fee_usd"]      for t in trades)
    realized_pnl   = sum(t["realized_pnl_usd"] for t in trades)

    pnls_on_sells = [t["realized_pnl_usd"] for t in sells if t["realized_pnl_usd"] != 0.0]
    wins   = [p for p in pnls_on_sells if p > 0]
    losses = [p for p in pnls_on_sells if p < 0]
    win_rate = (len(wins) / len(pnls_on_sells) * 100) if pnls_on_sells else 0.0

    by_asset: dict[str, dict] = defaultdict(lambda: {"buys": 0, "sells": 0, "notional": 0.0})
    for t in trades:
        by_asset[t["asset"]][t["side"] + "s"] += 1
        by_asset[t["asset"]]["notional"] += t["notional_usd"]

    print(f"  Session window : {first_et.strftime('%Y-%m-%d %H:%M:%S ET')} → "
          f"{last_et.strftime('%H:%M:%S ET')}")
    print(f"  Total trades   : {len(trades)}  ({len(buys)} buys, {len(sells)} sells)")
    print(f"  Total notional : ${total_notional:,.2f}")
    print(f"  Total fees     : ${total_fees:,.4f}")
    print(f"  Realized PnL   : {_fmt_usd(realized_pnl, signed=True)}")
    print(f"  Win rate       : {win_rate:.1f}%"
          f"  ({len(wins)} winners / {len(pnls_on_sells)} closed positions)")
    if wins:
        print(f"  Avg win        : {_fmt_usd(sum(wins)/len(wins), signed=True)}")
    if losses:
        print(f"  Avg loss       : {_fmt_usd(sum(losses)/len(losses), signed=True)}")
    print()

    print("  Per-asset breakdown:")
    print(f"  {'Asset':<6}  {'Buys':>5}  {'Sells':>5}  {'Notional':>12}")
    print("  " + "-" * 36)
    for asset, d in sorted(by_asset.items()):
        print(f"  {asset:<6}  {d['buys']:>5}  {d['sells']:>5}  ${d['notional']:>10,.2f}")
    print()

    print("  Trade-by-trade log:")
    print(f"  {'#':>4}  {'Time (ET)':>19}  {'Asset':<6}  {'Side':<4}  "
          f"{'Size':>12}  {'Price':>10}  {'Notional':>10}  {'Fee':>8}  {'rPnL':>10}")
    print("  " + "-" * 96)
    for i, t in enumerate(sorted(trades, key=lambda x: x["timestamp"]), 1):
        et = t["timestamp"].astimezone(ET)
        print(
            f"  {i:>4}  {et.strftime('%Y-%m-%d %H:%M:%S'):>19}  "
            f"{t['asset']:<6}  {t['side']:<4}  "
            f"{t['size_coins']:>12.6f}  {t['fill_price']:>10.4f}  "
            f"{t['notional_usd']:>10.2f}  {t['fee_usd']:>8.4f}  "
            f"{t['realized_pnl_usd']:>+10.4f}"
        )
    print()


def summarise_jsonl_trades(trades: list[dict], show_all: bool = False) -> None:
    if not trades:
        return

    live_trades = [t for t in trades if t["source"] == "live"]
    rl_trades   = [t for t in trades if t["source"] == "rl_agent"]

    if live_trades:
        first_ts = min(t["timestamp"] for t in live_trades)
        last_ts  = max(t["timestamp"] for t in live_trades)
        first_et = first_ts.astimezone(ET)
        last_et  = last_ts.astimezone(ET)

        buys  = [t for t in live_trades if t["side"] == "buy"]
        sells = [t for t in live_trades if t["side"] == "sell"]

        print("  ⚠️  LIVE ORDER HISTORY (real Kraken txids found in trades.jsonl)")
        print("  ─────────────────────────────────────────────────────────────")
        print(f"  These {len(live_trades)} trades were acknowledged by Kraken with real txids.")
        print(f"  This is from a PREVIOUS run of the bot (before PaperBroker was active).")
        print()
        print(f"  Session window : {first_et.strftime('%Y-%m-%d %H:%M:%S ET')} → "
              f"{last_et.strftime('%H:%M:%S ET')}")
        print(f"  Total trades   : {len(live_trades)}  ({len(buys)} buys, {len(sells)} sells)")
        print()
        print(f"  {'Asset':<6}  {'Buys':>5}  {'Sells':>5}")
        print("  " + "-" * 22)
        by_asset: dict[str, dict] = defaultdict(lambda: {"buys": 0, "sells": 0})
        for t in live_trades:
            by_asset[t["asset"]][t["side"] + "s"] += 1
        for asset, d in sorted(by_asset.items()):
            print(f"  {asset:<6}  {d['buys']:>5}  {d['sells']:>5}")
        print()

        if show_all:
            print("  All live trades:")
            print(f"  {'Time (ET)':>19}  {'Asset':<6}  {'Side':<5}  {'Size':>12}  {'txid'}")
            print("  " + "-" * 75)
            for t in sorted(live_trades, key=lambda x: x["timestamp"]):
                et = t["timestamp"].astimezone(ET)
                txid_str = ", ".join(t.get("txid", [])) or "—"
                print(f"  {et.strftime('%Y-%m-%d %H:%M:%S'):>19}  "
                      f"{t['asset']:<6}  {t['side']:<5}  {t['size_coins']:>12.6f}  {txid_str}")
            print()

    if rl_trades:
        buys  = [t for t in rl_trades if t["side"] == "buy"]
        sells = [t for t in rl_trades if t["side"] == "sell"]
        first_ts = min(t["timestamp"] for t in rl_trades)
        last_ts  = max(t["timestamp"] for t in rl_trades)
        print(f"  RL-agent trades (no txid — possibly validate=true responses): {len(rl_trades)}")
        print(f"  ({len(buys)} buys, {len(sells)} sells)")
        print()


# ---------------------------------------------------------------------------
# EOD report if present
# ---------------------------------------------------------------------------

def find_eod_report(logs_dir: str) -> str | None:
    """Return path to the most recent eod_YYYYMMDD.txt file, or None."""
    candidates = []
    if os.path.isdir(logs_dir):
        for name in os.listdir(logs_dir):
            if name.startswith("eod_") and name.endswith(".txt"):
                candidates.append(os.path.join(logs_dir, name))
    return max(candidates, default=None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    show_all = "--all" in sys.argv

    now_et = datetime.now(ET)
    now_utc = datetime.now(timezone.utc)

    print()
    print("=" * 70)
    print("  KrakBot — Session Performance Review")
    print(f"  Report generated: {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 70)
    print()

    # ── 1. Check for saved EOD report ──────────────────────────────────────
    eod_path = find_eod_report(LOGS_DIR)
    if eod_path:
        print(f"  📄  EOD report found: {eod_path}")
        print()
        print("─" * 70)
        with open(eod_path) as fh:
            print(fh.read())
        print("─" * 70)
        print()

    # ── 2. PaperBroker CSV ─────────────────────────────────────────────────
    paper_trades = load_paper_trades(CSV_PATH)
    print("─" * 70)
    print("  📊  PaperBroker fills  (project/logs/paper_trades.csv)")
    print("─" * 70)
    summarise_paper_trades(paper_trades)

    # ── 3. JSONL journal ───────────────────────────────────────────────────
    jsonl_trades = load_jsonl_trades(JSONL_PATH)
    if jsonl_trades:
        print("─" * 70)
        print("  📋  trades.jsonl  (RL-agent / LiveBroker journal)")
        print("─" * 70)
        summarise_jsonl_trades(jsonl_trades, show_all=show_all)

    # ── 4. What to watch next ──────────────────────────────────────────────
    # Market hours check
    weekday = now_et.weekday()
    mkt_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    mkt_close = now_et.replace(hour=16, minute=30, second=0, microsecond=0)

    if weekday < 5 and mkt_open <= now_et < mkt_close:
        status = "OPEN ✅  (ETP + crypto both active)"
    elif weekday < 5 and now_et < mkt_open:
        wait = int((mkt_open - now_et).total_seconds() / 60)
        status = f"Pre-market — opens in {wait}m"
    else:
        candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        wait = int((candidate - now_et).total_seconds() / 3600)
        status = f"After close — next open {candidate.strftime('%A')} in {wait}h"

    print("─" * 70)
    print(f"  🕐  Current time  : {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"  📈  Market status : {status}")
    print()
    print("  Log files:")
    print(f"    Paper trades CSV : {CSV_PATH}")
    print(f"    Trade journal    : {JSONL_PATH}")
    if eod_path:
        print(f"    EOD report       : {eod_path}")
    print()
    if not paper_trades and not jsonl_trades:
        print("  💡  No trades found. Run ./run_sandbox.sh to start the bot.")
    elif len(paper_trades) <= 1 and not eod_path:
        print("  💡  Only 1 paper trade found — the bot may still be starting up,")
        print("      or market conditions did not trigger entries yet today.")
        print("      Run ./run_sandbox.sh to start a fresh session.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
