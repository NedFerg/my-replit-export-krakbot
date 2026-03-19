"""
PortfolioSimulator — simulated order execution and P&L tracking.

Replaces Kraken order execution in backtest mode.  Tracks USD and asset
balances, applies fees and slippage, and records every trade.

Usage
-----
    sim = PortfolioSimulator(initial_usd=10_000, taker_fee=0.004)
    sim.set_price("BTC", 65_000)

    sim.buy("BTC", usd_amount=1_000)     # Buy $1 000 worth of BTC
    sim.sell("BTC", quantity=0.005)      # Sell 0.005 BTC

    equity = sim.get_total_equity({"BTC": 67_000})
    print(sim.get_summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single executed trade."""

    timestamp: datetime
    asset: str
    side: str          # "buy" | "sell"
    quantity: float
    price: float       # fill price (after slippage)
    fee_usd: float
    usd_spent: float   # positive = cash out; negative = cash in
    slippage_pct: float


class PortfolioSimulator:
    """
    Simulated portfolio tracker for backtesting.

    Parameters
    ----------
    initial_usd : float
        Starting cash balance in USD.
    assets : list[str] | None
        Optional list of asset symbols (e.g. ["BTC", "ETH"]).  Additional
        assets can be traded without pre-declaring them.
    taker_fee : float
        Fractional taker fee (e.g. 0.004 = 0.40 %).
    maker_fee : float
        Fractional maker fee (e.g. 0.0016 = 0.16 %).
    slippage : float
        Fractional one-way slippage applied to each fill
        (e.g. 0.001 = 0.10 %).
    """

    def __init__(
        self,
        initial_usd: float = 10_000.0,
        assets: list[str] | None = None,
        taker_fee: float = 0.004,
        maker_fee: float = 0.0016,
        slippage: float = 0.001,
    ) -> None:
        self.initial_usd = initial_usd
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage = slippage

        # Cash and asset balances
        self.usd_balance: float = initial_usd
        self.positions: dict[str, float] = {}   # asset → quantity held
        if assets:
            for a in assets:
                self.positions[a] = 0.0

        # Current prices (updated externally each candle)
        self._prices: dict[str, float] = {}

        # Trade history
        self.trades: list[Trade] = []

        # Equity curve: list of (timestamp, equity_usd) tuples
        self.equity_curve: list[tuple] = []

        # Realised P&L per asset: asset → cumulative USD gain/loss
        self._realized_pnl: dict[str, float] = {}

        # Average cost basis per asset (for unrealised P&L)
        self._avg_cost: dict[str, float] = {}

        logger.info(
            "PortfolioSimulator: initial_usd=%.2f  fee=%.2f%%  slippage=%.2f%%",
            initial_usd,
            taker_fee * 100,
            slippage * 100,
        )

    # ------------------------------------------------------------------
    # Price management
    # ------------------------------------------------------------------

    def set_price(self, asset: str, price: float) -> None:
        """Update the current price for *asset*."""
        self._prices[asset] = price

    def set_prices(self, prices: dict[str, float]) -> None:
        """Bulk-update prices from a dict."""
        self._prices.update(prices)

    def get_price(self, asset: str) -> float:
        """Return the current price for *asset* (raises if unknown)."""
        if asset not in self._prices:
            raise ValueError(f"PortfolioSimulator: no price set for {asset!r}")
        return self._prices[asset]

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def buy(
        self,
        asset: str,
        usd_amount: float | None = None,
        quantity: float | None = None,
        timestamp: datetime | None = None,
        use_maker: bool = False,
    ) -> Trade | None:
        """
        Buy *asset* using either a USD amount or a specific quantity.

        Parameters
        ----------
        asset      : asset symbol (e.g. "BTC")
        usd_amount : USD to spend (gross, before fee/slippage)
        quantity   : number of units to buy (alternative to usd_amount)
        timestamp  : candle timestamp (defaults to now)
        use_maker  : use maker fee instead of taker fee

        Returns
        -------
        Trade record, or None if the order cannot be filled.
        """
        if usd_amount is None and quantity is None:
            raise ValueError("buy(): supply either usd_amount or quantity")

        price = self.get_price(asset)
        fill_price = price * (1 + self.slippage)  # buy at slightly higher price
        fee_rate = self.maker_fee if use_maker else self.taker_fee

        if quantity is not None:
            usd_amount = quantity * fill_price

        if usd_amount is not None:
            fee_usd = usd_amount * fee_rate
            total_cost = usd_amount + fee_usd
            if total_cost > self.usd_balance:
                logger.warning(
                    "buy(%s): insufficient USD (need %.2f, have %.2f) — skipping",
                    asset, total_cost, self.usd_balance,
                )
                return None
            qty = usd_amount / fill_price

        ts = timestamp or datetime.now(tz=timezone.utc)

        # Apply balances
        self.usd_balance -= (usd_amount + fee_usd)
        self.positions[asset] = self.positions.get(asset, 0.0) + qty

        # Update average cost basis
        prev_qty = self.positions[asset] - qty
        prev_cost = self._avg_cost.get(asset, fill_price)
        if prev_qty > 0:
            new_cost = (prev_cost * prev_qty + fill_price * qty) / self.positions[asset]
        else:
            new_cost = fill_price
        self._avg_cost[asset] = new_cost

        trade = Trade(
            timestamp=ts,
            asset=asset,
            side="buy",
            quantity=qty,
            price=fill_price,
            fee_usd=fee_usd,
            usd_spent=usd_amount + fee_usd,
            slippage_pct=self.slippage * 100,
        )
        self.trades.append(trade)

        logger.debug(
            "BUY  %s  qty=%.6f  @ %.2f  fee=%.2f  cash_after=%.2f",
            asset, qty, fill_price, fee_usd, self.usd_balance,
        )
        return trade

    def sell(
        self,
        asset: str,
        quantity: float | None = None,
        usd_target: float | None = None,
        timestamp: datetime | None = None,
        use_maker: bool = False,
    ) -> Trade | None:
        """
        Sell *asset* by quantity or USD target.

        Parameters
        ----------
        asset      : asset symbol
        quantity   : units to sell (defaults to entire position)
        usd_target : sell enough units to receive this USD amount (gross)
        timestamp  : candle timestamp
        use_maker  : use maker fee

        Returns
        -------
        Trade record, or None if the order cannot be filled.
        """
        price = self.get_price(asset)
        fill_price = price * (1 - self.slippage)  # sell at slightly lower price
        fee_rate = self.maker_fee if use_maker else self.taker_fee

        held = self.positions.get(asset, 0.0)
        if held <= 0:
            logger.warning("sell(%s): no position to sell — skipping", asset)
            return None

        if usd_target is not None:
            quantity = usd_target / fill_price

        if quantity is None:
            quantity = held

        quantity = min(quantity, held)  # cannot sell more than we hold

        gross_usd = quantity * fill_price
        fee_usd = gross_usd * fee_rate
        net_usd = gross_usd - fee_usd

        ts = timestamp or datetime.now(tz=timezone.utc)

        # Apply balances
        self.positions[asset] = held - quantity
        self.usd_balance += net_usd

        # Realised P&L
        cost_basis = self._avg_cost.get(asset, fill_price)
        realised = (fill_price - cost_basis) * quantity - fee_usd
        self._realized_pnl[asset] = self._realized_pnl.get(asset, 0.0) + realised

        trade = Trade(
            timestamp=ts,
            asset=asset,
            side="sell",
            quantity=quantity,
            price=fill_price,
            fee_usd=fee_usd,
            usd_spent=-(net_usd),   # negative = cash in
            slippage_pct=self.slippage * 100,
        )
        self.trades.append(trade)

        logger.debug(
            "SELL %s  qty=%.6f  @ %.2f  fee=%.2f  cash_after=%.2f  realised=%.2f",
            asset, quantity, fill_price, fee_usd, self.usd_balance, realised,
        )
        return trade

    def close_position(self, asset: str, timestamp: datetime | None = None) -> Trade | None:
        """Sell the entire position in *asset*."""
        return self.sell(asset, timestamp=timestamp)

    def rebalance(
        self,
        target_weights: dict[str, float],
        timestamp: datetime | None = None,
    ) -> list[Trade]:
        """
        Rebalance portfolio to *target_weights* (fractions of total equity).

        Parameters
        ----------
        target_weights : dict mapping asset → target fraction of equity
            e.g. {"BTC": 0.6, "ETH": 0.3}  (remainder stays in USD)
        """
        prices = dict(self._prices)
        equity = self.get_total_equity(prices)
        if equity <= 0:
            return []

        executed: list[Trade] = []
        ts = timestamp or datetime.now(tz=timezone.utc)

        # First pass: sell assets that need to decrease
        for asset, target_w in target_weights.items():
            if asset not in prices:
                continue
            target_usd = equity * target_w
            current_usd = self.positions.get(asset, 0.0) * prices[asset]
            delta = target_usd - current_usd
            if delta < -1.0:  # need to sell
                qty = abs(delta) / prices[asset]
                t = self.sell(asset, quantity=qty, timestamp=ts)
                if t:
                    executed.append(t)

        # Second pass: buy assets that need to increase
        for asset, target_w in target_weights.items():
            if asset not in prices:
                continue
            target_usd = equity * target_w
            current_usd = self.positions.get(asset, 0.0) * prices[asset]
            delta = target_usd - current_usd
            if delta > 1.0:  # need to buy
                t = self.buy(asset, usd_amount=delta, timestamp=ts)
                if t:
                    executed.append(t)

        return executed

    # ------------------------------------------------------------------
    # Portfolio metrics
    # ------------------------------------------------------------------

    def get_total_equity(self, prices: dict[str, float] | None = None) -> float:
        """
        Return total portfolio value in USD.

        Parameters
        ----------
        prices : optional price override dict; defaults to internal prices.
        """
        p = prices or self._prices
        asset_value = sum(
            self.positions.get(a, 0.0) * p[a]
            for a in p
            if a in self.positions
        )
        return self.usd_balance + asset_value

    def get_realized_pnl(self) -> float:
        """Return total realised P&L across all assets (USD)."""
        return sum(self._realized_pnl.values())

    def get_unrealized_pnl(self, prices: dict[str, float] | None = None) -> float:
        """Return total unrealised P&L across all open positions (USD)."""
        p = prices or self._prices
        total = 0.0
        for asset, qty in self.positions.items():
            if qty > 0 and asset in p:
                cost = self._avg_cost.get(asset, p[asset])
                total += (p[asset] - cost) * qty
        return total

    def get_position_value(self, asset: str, prices: dict[str, float] | None = None) -> float:
        """Return current USD value of position in *asset*."""
        p = prices or self._prices
        qty = self.positions.get(asset, 0.0)
        price = p.get(asset, 0.0)
        return qty * price

    def get_exposure_pct(self, asset: str, prices: dict[str, float] | None = None) -> float:
        """Return *asset* position as a fraction of total equity (0-1)."""
        equity = self.get_total_equity(prices)
        if equity == 0:
            return 0.0
        return self.get_position_value(asset, prices) / equity

    def record_equity(self, timestamp: datetime, prices: dict[str, float] | None = None) -> float:
        """
        Record a snapshot of total equity to the equity curve.

        Returns the recorded equity value.
        """
        equity = self.get_total_equity(prices)
        self.equity_curve.append((timestamp, equity))
        return equity

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_summary(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        """Return a summary dict of the current portfolio state."""
        p = prices or self._prices
        equity = self.get_total_equity(p)
        return {
            "usd_balance": round(self.usd_balance, 2),
            "total_equity": round(equity, 2),
            "total_return_pct": round((equity - self.initial_usd) / self.initial_usd * 100, 2),
            "realized_pnl": round(self.get_realized_pnl(), 2),
            "unrealized_pnl": round(self.get_unrealized_pnl(p), 2),
            "num_trades": len(self.trades),
            "positions": {
                a: {"quantity": round(q, 8), "value_usd": round(q * p.get(a, 0), 2)}
                for a, q in self.positions.items()
                if q > 0
            },
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Return all trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "asset": t.asset,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "fee_usd": t.fee_usd,
                "usd_spent": t.usd_spent,
                "slippage_pct": t.slippage_pct,
            }
            for t in self.trades
        ])

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Return equity curve as a DataFrame with columns [timestamp, equity]."""
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])
        return pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])

    def reset(self) -> None:
        """Reset the simulator to its initial state."""
        self.usd_balance = self.initial_usd
        self.positions.clear()
        self._prices.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._realized_pnl.clear()
        self._avg_cost.clear()

    def __repr__(self) -> str:
        equity = self.get_total_equity()
        ret = (equity - self.initial_usd) / self.initial_usd * 100
        return (
            f"PortfolioSimulator(equity=${equity:,.2f}  "
            f"return={ret:+.1f}%  trades={len(self.trades)})"
        )
