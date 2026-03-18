import { Router, type IRouter, type Request, type Response } from "express";
import { DatabaseSync } from "node:sqlite";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import {
  ListTradesQueryParams,
  ListTradesResponse,
  GetPerformanceQueryParams,
  GetPerformanceResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();

// ---------------------------------------------------------------------------
// Database path — override via TRADE_ARCHIVE_PATH env var, otherwise resolve
// relative to the repo root (two levels up from artifacts/api-server/src).
// ---------------------------------------------------------------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_ARCHIVE = path.resolve(
  __dirname,
  "../../../../project/logs/trade_archive.db",
);
const ARCHIVE_PATH = process.env["TRADE_ARCHIVE_PATH"] ?? DEFAULT_ARCHIVE;

function openDb(): DatabaseSync | null {
  if (!fs.existsSync(ARCHIVE_PATH)) {
    return null;
  }
  try {
    return new DatabaseSync(ARCHIVE_PATH, { readOnly: true });
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// GET /api/trades
// ---------------------------------------------------------------------------
router.get("/trades", (req: Request, res: Response) => {
  const parsed = ListTradesQueryParams.safeParse(req.query);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.flatten() });
    return;
  }

  const { limit, asset, strategy } = parsed.data;

  const db = openDb();
  if (!db) {
    const result = ListTradesResponse.parse({ trades: [], total: 0 });
    res.json(result);
    return;
  }

  try {
    // Build WHERE clauses
    const conditions: string[] = [];
    const params: (string | number)[] = [];

    if (asset) {
      conditions.push("asset = ?");
      params.push(asset);
    }
    if (strategy) {
      conditions.push("strategy_name = ?");
      params.push(strategy);
    }

    const where =
      conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    // Total count
    const countStmt = db.prepare(`SELECT COUNT(*) AS n FROM trades ${where}`);
    const countRow = countStmt.get(...params) as { n: number } | undefined;
    const total = countRow?.n ?? 0;

    // Fetch rows newest-first
    params.push(limit);
    const rowsStmt = db.prepare(
      `SELECT * FROM trades ${where} ORDER BY timestamp DESC LIMIT ?`,
    );
    const rows = rowsStmt.all(...params);

    const result = ListTradesResponse.parse({ trades: rows, total });
    res.json(result);
  } finally {
    db.close();
  }
});

// ---------------------------------------------------------------------------
// GET /api/performance
// ---------------------------------------------------------------------------
router.get("/performance", (req: Request, res: Response) => {
  const parsed = GetPerformanceQueryParams.safeParse(req.query);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.flatten() });
    return;
  }

  const { period } = parsed.data;

  const db = openDb();
  if (!db) {
    const result = GetPerformanceResponse.parse({
      period,
      trade_count: 0,
      total_notional_usd: 0,
      total_fees_usd: 0,
      total_realized_pnl_usd: 0,
      win_rate: 0,
      avg_return_pct: 0,
      cumulative_volume_usd: 0,
    });
    res.json(result);
    return;
  }

  try {
    // Build period filter
    let dateFilter = "";
    if (period === "daily") {
      dateFilter = `WHERE timestamp >= date('now')`;
    } else if (period === "weekly") {
      dateFilter = `WHERE timestamp >= date('now', '-7 days')`;
    } else if (period === "monthly") {
      dateFilter = `WHERE timestamp >= date('now', '-30 days')`;
    }

    const stmt = db.prepare(`
      SELECT
        COUNT(*)                                          AS trade_count,
        COALESCE(SUM(notional_usd), 0)                   AS total_notional_usd,
        COALESCE(SUM(fee_usd), 0)                        AS total_fees_usd,
        COALESCE(SUM(realized_pnl_usd), 0)              AS total_realized_pnl_usd,
        COALESCE(AVG(return_pct), 0)                     AS avg_return_pct,
        COALESCE(MAX(cumulative_volume), 0)              AS cumulative_volume_usd,
        COALESCE(
          1.0 * SUM(CASE WHEN side = 'sell' AND realized_pnl_usd > 0 THEN 1 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END), 0),
          0
        )                                                AS win_rate
      FROM trades
      ${dateFilter}
    `);

    const row = stmt.get() as {
      trade_count: number;
      total_notional_usd: number;
      total_fees_usd: number;
      total_realized_pnl_usd: number;
      avg_return_pct: number;
      cumulative_volume_usd: number;
      win_rate: number;
    };

    const result = GetPerformanceResponse.parse({
      period,
      trade_count: row.trade_count,
      total_notional_usd: row.total_notional_usd,
      total_fees_usd: row.total_fees_usd,
      total_realized_pnl_usd: row.total_realized_pnl_usd,
      win_rate: row.win_rate,
      avg_return_pct: row.avg_return_pct,
      cumulative_volume_usd: row.cumulative_volume_usd,
    });

    res.json(result);
  } finally {
    db.close();
  }
});

export default router;
