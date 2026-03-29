#!/usr/bin/env python3
"""
recover_trades.py — Trade Log Recovery & Merge CLI Tool

PURPOSE
-------
If the local trade log (paper_trades.csv or live_trades.csv) was lost or
corrupted, this script helps you:

  1. Detect and rotate any file that still has git merge-conflict markers.
  2. Merge trades from a manually exported Kraken CSV (or any compatible CSV)
     into the current log without creating duplicate rows.
  3. Merge any `.fallback` backup rows (auto-created when the main log was
     temporarily unwritable) back into the primary log.

USAGE
-----
# Audit and auto-rotate a conflicted log:
    python scripts/recover_trades.py audit project/logs/paper_trades.csv

# Merge a Kraken export into the paper trade log (deduplicated):
    python scripts/recover_trades.py merge \
        --source kraken_export.csv \
        --target project/logs/paper_trades.csv

# Merge a .fallback file back into the primary log:
    python scripts/recover_trades.py merge \
        --source project/logs/paper_trades.csv.fallback \
        --target project/logs/paper_trades.csv

# Merge a .fallback file back into the live log:
    python scripts/recover_trades.py merge \
        --source project/logs/live_trades.csv.fallback \
        --target project/logs/live_trades.csv

DEDUPLICATION
-------------
Rows with an identical (timestamp, asset, side) triplet are considered
duplicates.  Only the first occurrence of each triplet is kept so that
re-importing the same export multiple times is safe.

EXIT CODES
----------
  0 — success (no conflicts, or merge completed)
  1 — conflict markers found and rotated (re-run to confirm clean state)
  2 — argument or file error
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# Make sure the project package is importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from broker.broker import _recover_csv_conflict_markers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOG_HEADERS = [
    "timestamp", "asset", "side", "size_coins", "fill_price",
    "notional_usd", "fee_usd", "realized_pnl_usd", "position_after_trade",
]


def _read_rows(path: str) -> list[dict]:
    """Read a CSV file and return a list of row dicts.  Skips bad rows."""
    rows: list[dict] = []
    try:
        with open(path, newline="", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except OSError as exc:
        print(f"[ERROR] Cannot read {path!r}: {exc}", file=sys.stderr)
    return rows


def _write_rows(path: str, headers: list[str], rows: list[dict]) -> None:
    """Write a list of row dicts to a CSV file (overwrites)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _dedup_key(row: dict) -> tuple:
    """Return a deduplication key for a trade row."""
    return (row.get("timestamp", ""), row.get("asset", ""), row.get("side", ""))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_audit(path: str) -> int:
    """Audit a log file and rotate it if conflict markers are found."""
    print(f"[audit] Checking {path!r} for git merge conflict markers …")
    bak = _recover_csv_conflict_markers(path)
    if bak:
        print(
            f"\n[audit] !! Conflict markers found !!\n"
            f"  Corrupted file backed up to: {bak}\n"
            f"  A fresh log will be created on next bot startup.\n"
            f"  Inspect the backup and use the 'merge' subcommand to\n"
            f"  import any valid rows back into a clean log."
        )
        return 1
    print(f"[audit] {path!r} is clean — no action needed.")
    return 0


def cmd_merge(source: str, target: str) -> int:
    """Merge rows from *source* into *target*, deduplicating on (ts, asset, side)."""
    if not os.path.exists(source):
        print(f"[ERROR] Source file not found: {source!r}", file=sys.stderr)
        return 2

    # Audit target first
    bak = _recover_csv_conflict_markers(target)
    if bak:
        print(
            f"[merge] WARNING: target had conflict markers — rotated to {bak}.\n"
            f"  Merging source into a freshly created target."
        )

    # Read existing rows from target (may not exist yet)
    existing = _read_rows(target) if os.path.exists(target) else []
    existing_keys = {_dedup_key(r) for r in existing}
    print(f"[merge] Target {target!r}: {len(existing)} existing rows.")

    # Read source rows
    new_rows = _read_rows(source)
    print(f"[merge] Source {source!r}: {len(new_rows)} rows to process.")

    # Filter: only rows whose key is not already present
    added = [r for r in new_rows if _dedup_key(r) not in existing_keys]
    skipped = len(new_rows) - len(added)
    print(f"[merge] Skipped {skipped} duplicate rows.  Adding {len(added)} new rows.")

    if not added:
        print("[merge] Nothing to merge — target is already up-to-date.")
        return 0

    # Merge and write sorted by timestamp
    all_rows = existing + added
    all_rows.sort(key=lambda r: r.get("timestamp", ""))

    # Determine headers: use the standard set if the source has at least one
    # standard column; fall back to the source's own column names otherwise
    # (e.g. a raw Kraken export with non-standard column names).
    headers = _LOG_HEADERS
    if new_rows and not any(h in new_rows[0] for h in _LOG_HEADERS):
        headers = list(new_rows[0].keys())
        print(
            f"[merge] NOTE: source columns do not match the standard log headers.\n"
            f"  Source columns : {headers}\n"
            f"  Standard headers: {_LOG_HEADERS}\n"
            f"  Writing with source columns — review the merged file manually."
        )

    # Back up the existing target before overwriting
    if os.path.exists(target):
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        pre_merge_bak = f"{target}.pre_merge_{ts}"
        os.replace(target, pre_merge_bak)
        print(f"[merge] Backed up existing target to {pre_merge_bak!r}.")

    _write_rows(target, headers, all_rows)
    print(
        f"[merge] Done.  {target!r} now has {len(all_rows)} rows "
        f"({len(added)} new from {source!r})."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trade log recovery and merge tool for Krakbot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # audit sub-command
    p_audit = sub.add_parser("audit", help="Check a log file for conflict markers and rotate if found.")
    p_audit.add_argument("path", help="Path to the CSV log file to audit.")

    # merge sub-command
    p_merge = sub.add_parser(
        "merge",
        help="Merge rows from a source CSV into the target log (deduplicates).",
    )
    p_merge.add_argument("--source", required=True, help="Source CSV to import rows from.")
    p_merge.add_argument("--target", required=True, help="Target log file to merge rows into.")

    args = parser.parse_args(argv)

    if args.command == "audit":
        return cmd_audit(args.path)
    if args.command == "merge":
        return cmd_merge(args.source, args.target)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
