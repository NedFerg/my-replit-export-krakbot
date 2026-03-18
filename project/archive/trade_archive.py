"""
Trade archive — persistent SQLite store for paper/live fills.

Every fill recorded via record_trade() is inserted into a local SQLite
database (project/logs/trade_archive.db).  The archive is designed to:

  · survive bot restarts (persistent audit trail)
  · support analytics queries (win rate, Sharpe, drawdown, fee totals)
  · track cumulative 30-day volume so fee tiers can be computed accurately
  · attribute each trade to the strategy that generated it

Typical usage
-------------
    from archive.trade_archive import TradeArchive

    archive = TradeArchive()                    # auto-creates DB on first use
    archive.record_trade(record, strategy_name="RLTrader")
    summary = archive.performance_summary(period="monthly")
    trades  = archive.get_trades(min_return=-0.01)
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Default database path (relative to this file)
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "logs", "trade_archive.db"
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id            TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    asset               TEXT NOT NULL,
    side                TEXT NOT NULL,
    size_coins          REAL NOT NULL,
    fill_price          REAL NOT NULL,
    notional_usd        REAL NOT NULL,
    fee_usd             REAL NOT NULL,
    realized_pnl_usd    REAL NOT NULL,
    position_after_trade REAL NOT NULL,
    strategy_name       TEXT NOT NULL DEFAULT '',
    return_pct          REAL NOT NULL DEFAULT 0.0,
    cumulative_volume   REAL NOT NULL DEFAULT 0.0
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_asset     ON trades (asset);
CREATE INDEX IF NOT EXISTS idx_trades_strategy  ON trades (strategy_name);
"""


class TradeArchive:
    """
    Persistent SQLite archive for paper-broker fills.

    Parameters
    ----------
    db_path : Path to the SQLite database file.  Auto-created on first use.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = os.path.abspath(db_path or _DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        print(f"[TradeArchive] SQLite archive opened: {self.db_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist yet."""
        cur = self._conn.cursor()
        cur.executescript(_CREATE_TABLE_SQL + _CREATE_INDEX_SQL)
        self._conn.commit()

    def _cumulative_volume(self) -> float:
        """Return the total notional_usd ever recorded in the archive."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(notional_usd), 0.0) FROM trades"
        ).fetchone()
        return float(row[0])

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        record: dict[str, Any],
        strategy_name: str = "",
    ) -> str:
        """
        Persist one fill to the archive.

        Parameters
        ----------
        record        : Dict from PaperBroker._paper_fill — must contain:
                        timestamp, asset, side, size_coins, fill_price,
                        notional_usd, fee_usd, realized_pnl_usd,
                        position_after_trade.
        strategy_name : Name of the strategy/agent that generated the trade.

        Returns
        -------
        str — the generated trade_id (UUID4).
        """
        notional = float(record.get("notional_usd", 0.0))
        fee      = float(record.get("fee_usd", 0.0))

        # return_pct = realized_pnl / (notional + fee) to avoid div-by-zero
        denom = notional + fee
        return_pct = (
            float(record.get("realized_pnl_usd", 0.0)) / denom
            if denom > 0 else 0.0
        )

        cumvol = self._cumulative_volume() + notional
        trade_id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT INTO trades (
                trade_id, timestamp, asset, side, size_coins,
                fill_price, notional_usd, fee_usd, realized_pnl_usd,
                position_after_trade, strategy_name, return_pct, cumulative_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                str(record.get("timestamp", "")),
                str(record.get("asset", "")),
                str(record.get("side", "")),
                float(record.get("size_coins", 0.0)),
                float(record.get("fill_price", 0.0)),
                notional,
                fee,
                float(record.get("realized_pnl_usd", 0.0)),
                float(record.get("position_after_trade", 0.0)),
                strategy_name,
                return_pct,
                cumvol,
            ),
        )
        self._conn.commit()
        return trade_id

    # ------------------------------------------------------------------
    # Read / query API
    # ------------------------------------------------------------------

    def get_trades(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        strategy: str | None = None,
        asset: str | None = None,
        min_return: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return trades matching the given filters.

        Parameters
        ----------
        start_date : ISO-8601 string (inclusive).  E.g. "2026-03-01T00:00:00Z"
        end_date   : ISO-8601 string (inclusive).
        strategy   : Exact strategy_name match.
        asset      : Exact asset match (e.g. "SOL").
        min_return : Only include trades where return_pct >= min_return.

        Returns
        -------
        list of dicts, one per trade.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if start_date:
            clauses.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("timestamp <= ?")
            params.append(end_date)
        if strategy:
            clauses.append("strategy_name = ?")
            params.append(strategy)
        if asset:
            clauses.append("asset = ?")
            params.append(asset)
        if min_return is not None:
            clauses.append("return_pct >= ?")
            params.append(min_return)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM trades {where} ORDER BY timestamp ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def performance_summary(self, period: str = "all") -> dict[str, Any]:
        """
        Aggregate performance statistics.

        Parameters
        ----------
        period : One of "all", "daily", "weekly", "monthly".
                 When not "all", restricts to the most recent complete period.

        Returns
        -------
        dict with keys:
            trade_count, total_notional, total_fees, total_realized_pnl,
            win_rate, avg_return_pct, cumulative_volume, period_start, period_end.
        """
        now = datetime.now(timezone.utc)

        if period == "daily":
            period_start = (now - timedelta(days=1)).isoformat()
        elif period == "weekly":
            period_start = (now - timedelta(weeks=1)).isoformat()
        elif period == "monthly":
            period_start = (now - timedelta(days=30)).isoformat()
        else:
            period_start = None

        clauses = []
        params: list[Any] = []
        if period_start:
            clauses.append("timestamp >= ?")
            params.append(period_start)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        row = self._conn.execute(
            f"""
            SELECT
                COUNT(*)                            AS trade_count,
                COALESCE(SUM(notional_usd), 0.0)    AS total_notional,
                COALESCE(SUM(fee_usd), 0.0)         AS total_fees,
                COALESCE(SUM(realized_pnl_usd), 0.0) AS total_realized_pnl,
                COALESCE(AVG(return_pct), 0.0)      AS avg_return_pct,
                COALESCE(MAX(cumulative_volume), 0.0) AS cumulative_volume,
                MIN(timestamp)                      AS period_start,
                MAX(timestamp)                      AS period_end
            FROM trades {where}
            """,
            params,
        ).fetchone()

        # Win rate: fraction of sell trades with positive realized PnL
        sell_clauses = clauses + ["side = 'sell'"]
        pnl_clauses  = clauses + ["side = 'sell'", "realized_pnl_usd > 0"]
        wins_sql = f"SELECT COUNT(*) FROM trades WHERE {' AND '.join(pnl_clauses)}"
        sell_sql = f"SELECT COUNT(*) FROM trades WHERE {' AND '.join(sell_clauses)}"
        wins        = int(self._conn.execute(wins_sql, params).fetchone()[0])
        total_sells = int(self._conn.execute(sell_sql, params).fetchone()[0])
        win_rate    = wins / total_sells if total_sells > 0 else 0.0

        return {
            "trade_count":       int(row["trade_count"]),
            "total_notional":    round(float(row["total_notional"]), 4),
            "total_fees":        round(float(row["total_fees"]), 4),
            "total_realized_pnl": round(float(row["total_realized_pnl"]), 4),
            "win_rate":          round(win_rate, 4),
            "avg_return_pct":    round(float(row["avg_return_pct"]), 6),
            "cumulative_volume": round(float(row["cumulative_volume"]), 4),
            "period_start":      row["period_start"],
            "period_end":        row["period_end"],
        }

    def get_fee_tier_for_date(self, date: str) -> float:
        """
        Return the Kraken taker fee rate that was in effect at the given date,
        based on the cumulative_volume recorded just before that date.

        Parameters
        ----------
        date : ISO-8601 string, e.g. "2026-03-16T12:00:00Z".

        Returns
        -------
        float — taker fee as a decimal fraction.
        """
        # Import here to avoid a circular dependency at module load time
        from broker.broker import LiveBroker  # noqa: PLC0415

        row = self._conn.execute(
            "SELECT cumulative_volume FROM trades WHERE timestamp <= ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (date,),
        ).fetchone()

        volume = float(row["cumulative_volume"]) if row else 0.0
        return LiveBroker.get_kraken_taker_fee(volume)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "TradeArchive":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
