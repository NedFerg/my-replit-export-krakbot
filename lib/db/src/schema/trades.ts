import { pgTable, text, real, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

// ---------------------------------------------------------------------------
// Paper trades — mirrors the SQLite schema written by project/archive/
//   trade_archive.py so that trade history can be persisted in PostgreSQL
//   for longer-term analytics and cross-session queries.
// ---------------------------------------------------------------------------
export const tradesTable = pgTable("trades", {
  trade_id: text("trade_id").primaryKey(),
  timestamp: timestamp("timestamp", { withTimezone: true }).notNull(),
  asset: text("asset").notNull(),
  side: text("side", { enum: ["buy", "sell"] }).notNull(),
  size_coins: real("size_coins").notNull(),
  fill_price: real("fill_price").notNull(),
  notional_usd: real("notional_usd").notNull(),
  fee_usd: real("fee_usd").notNull(),
  realized_pnl_usd: real("realized_pnl_usd").notNull().default(0),
  position_after_trade: real("position_after_trade").notNull().default(0),
  strategy_name: text("strategy_name").notNull().default(""),
  return_pct: real("return_pct").notNull().default(0),
  cumulative_volume: real("cumulative_volume").notNull().default(0),
});

export const insertTradeSchema = createInsertSchema(tradesTable, {
  trade_id: z.string().uuid(),
  side: z.enum(["buy", "sell"]),
});

export type Trade = typeof tradesTable.$inferSelect;
export type NewTrade = typeof tradesTable.$inferInsert;
