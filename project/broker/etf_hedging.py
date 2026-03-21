"""
project/broker/etf_hedging.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ETF hedging and amplification strategy layer for Krakbot.

This module implements a dual-mode ETF overlay that sits on top of the
existing 24/7 crypto spot trading layer:

    Hedging mode     : SHORT ETFs (SETH, ETHD) activated during bearish regimes.
    Amplification    : LONG  ETFs (XXRP, SLON, ETHU) activated during bullish regimes.
    Neutral / Closed : ETF layer flattened to zero to minimise fees.

Key design constraints
----------------------
- Total ETF notional ≤ 30 % of portfolio equity  (``ETF_TOTAL_CAP`` env var).
- Hedging and amplification are MUTUALLY EXCLUSIVE — never held simultaneously.
- Orders respect US market hours via ``project/utils/market_hours``:
    * Regular hours → market orders
    * Premarket / after-hours → limit orders only
    * Closed → no ETF orders
- Crypto spot positions (24/7) are never modified by this module.

ETF universe
------------
Short ETFs (bearish hedge):
    SETH   3× short Ethereum
    ETHD   3× short Ethereum (alternative)

Long ETFs (bullish amplification):
    XXRP   3× long XRP
    SLON   3× long SOL
    ETHU   3× long ETH

Environment variables (all optional — defaults shown)
------------------------------------------------------
    ENABLE_ETF_TRADING   true
    ETF_HEDGE_CAP        0.30   max fraction of equity for short hedges
    ETF_AMPLIFY_CAP      0.30   max fraction of equity for long amplifiers
    ETF_TOTAL_CAP        0.30   combined hedge + amplify cap
    LIMIT_ORDER_TOLERANCE 0.001  limit price offset from mid (0.1 %)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.market_hours import (
    is_etf_tradeable,
    get_etf_order_type,
    get_market_period,
    MarketPeriod,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
)


# ---------------------------------------------------------------------------
# ETF asset catalogues
# ---------------------------------------------------------------------------

#: Short ETFs used for bearish hedging.  Ethereum leads the altcoin market
#: so shorting either SETH or ETHD provides broad-market downside protection.
SHORT_ETFS: List[str] = ["SETH", "ETHD"]

#: Long ETFs used for bullish amplification.
LONG_ETFS: List[str] = ["XXRP", "SLON", "ETHU"]

#: All ETF assets tracked by this layer.
ALL_ETFS: List[str] = SHORT_ETFS + LONG_ETFS

#: Kraken ticker symbols for each ETF.
#: Update if Kraken changes pair names.
ETF_KRAKEN_PAIRS: Dict[str, str] = {
    "SETH": "SETHUSD",
    "ETHD": "ETHDUSD",
    "XXRP": "XXRPUSD",
    "SLON": "SLONUSD",
    "ETHU": "ETHUUSD",
}

#: Default maximum fractional allocation per individual ETF (of total equity).
#: Each ETF gets at most half of the total cap so two ETFs in the same mode
#: together fill the cap cleanly.
ETF_DEFAULT_MAX_ALLOC: Dict[str, float] = {
    "SETH": 0.15,
    "ETHD": 0.15,
    "XXRP": 0.10,
    "SLON": 0.10,
    "ETHU": 0.10,
}


# ---------------------------------------------------------------------------
# Regime classification result
# ---------------------------------------------------------------------------

class ETFMode:
    """ETF layer operating mode."""
    HEDGE   = "hedge"       # short ETFs active
    AMPLIFY = "amplify"     # long ETFs active
    FLAT    = "flat"        # all positions closed


# ---------------------------------------------------------------------------
# ETF order descriptor
# ---------------------------------------------------------------------------

@dataclass
class ETFOrder:
    """A single ETF order ready for submission to the broker."""
    asset:      str          # e.g. "SETH"
    kraken_pair: str         # e.g. "SETHUSD"
    side:       str          # "buy" or "sell"
    order_type: str          # "mkt" or "limit"
    coin_units: float        # unsigned quantity in base-asset units
    limit_price: Optional[float] = None   # only set when order_type == "limit"
    reason:     str = ""     # human-readable rationale for logging


# ---------------------------------------------------------------------------
# ETF hedging layer
# ---------------------------------------------------------------------------

class ETFHedgingLayer:
    """
    Stateful ETF overlay manager.

    Usage
    -----
    1. Instantiate once and attach to the broker / agent.
    2. Call ``determine_mode(regime)`` each cycle to decide hedge/amplify/flat.
    3. Call ``build_orders(mode, equity, prices)`` to get a list of ``ETFOrder``
       objects respecting the 30 % cap and market hours.
    4. Submit orders through the broker's existing ``_submit_spot_order`` path
       (ETF orders are spot orders on Kraken, not futures).

    Parameters
    ----------
    total_cap       : Combined ETF notional cap as a fraction of equity (0.30).
    hedge_cap       : Sub-cap for short-ETF positions (≤ total_cap).
    amplify_cap     : Sub-cap for long-ETF positions (≤ total_cap).
    tolerance       : Limit-order price offset from mid (0.001 = 0.1 %).
    """

    def __init__(
        self,
        total_cap:   float = float(os.getenv("ETF_TOTAL_CAP",   "0.30")),
        hedge_cap:   float = float(os.getenv("ETF_HEDGE_CAP",   "0.30")),
        amplify_cap: float = float(os.getenv("ETF_AMPLIFY_CAP", "0.30")),
        tolerance:   float = float(os.getenv("LIMIT_ORDER_TOLERANCE", "0.001")),
    ):
        self.total_cap   = min(total_cap, 0.30)    # hard-ceiling at 30 %
        self.hedge_cap   = min(hedge_cap, total_cap)
        self.amplify_cap = min(amplify_cap, total_cap)
        self.tolerance   = tolerance

        # Enabled via env var (default true)
        _enabled = os.getenv("ENABLE_ETF_TRADING", "true").strip().lower()
        self.enabled: bool = _enabled not in ("false", "0", "no")

        # Current positions: asset → signed coin quantity.
        # Positive = long, negative = short (though ETFs themselves are not
        # sold short — negative here means we hold a short-ETF like SETH).
        self.etf_positions: Dict[str, float] = {etf: 0.0 for etf in ALL_ETFS}

        # Track last mode to detect regime transitions
        self._last_mode: str = ETFMode.FLAT

        if self.enabled:
            print(
                f"[ETF LAYER] Enabled — total_cap={self.total_cap:.0%}  "
                f"hedge_cap={self.hedge_cap:.0%}  "
                f"amplify_cap={self.amplify_cap:.0%}  "
                f"tolerance={self.tolerance:.3%}"
            )
        else:
            print("[ETF LAYER] Disabled via ENABLE_ETF_TRADING=false")

    # ------------------------------------------------------------------
    # Mode determination
    # ------------------------------------------------------------------

    def determine_mode(self, regime: dict) -> str:
        """
        Classify the current market regime into an ETF operating mode.

        Parameters
        ----------
        regime : dict with optional keys:
            bullish_confidence  float  0–1  (> 0.6 → amplify)
            panic_risk          int    0–2  (≥ 1  → hedge)
            bearish_drift       bool         (True → hedge)
            macro_regime        float  −1/0/+1  (−1 → hedge, +1 → amplify)

        Returns
        -------
        ETFMode.HEDGE | ETFMode.AMPLIFY | ETFMode.FLAT
        """
        if not self.enabled:
            return ETFMode.FLAT

        bullish_conf = float(regime.get("bullish_confidence", 0.0))
        panic_risk   = int(regime.get("panic_risk",           0))
        bearish_drift = bool(regime.get("bearish_drift",      False))
        macro_regime  = float(regime.get("macro_regime",      0.0))

        # Bearish signals take priority — hedge first, amplify never simultaneously
        if panic_risk >= 1 or bearish_drift or macro_regime <= -0.5:
            mode = ETFMode.HEDGE
        elif bullish_conf > 0.6 or macro_regime >= 0.5:
            mode = ETFMode.AMPLIFY
        else:
            mode = ETFMode.FLAT

        if mode != self._last_mode:
            print(f"[ETF LAYER] Regime transition: {self._last_mode} → {mode}")
        self._last_mode = mode
        return mode

    # ------------------------------------------------------------------
    # ETF notional accounting
    # ------------------------------------------------------------------

    def current_etf_notional(self, prices: Dict[str, float], equity: float) -> float:
        """
        Return the current total ETF notional as a fraction of equity.

        ``sum(|position_i × price_i|) / equity``
        """
        if equity <= 0:
            return 0.0
        total = sum(
            abs(self.etf_positions.get(etf, 0.0)) * prices.get(etf, 0.0)
            for etf in ALL_ETFS
        )
        return total / equity

    def _notional_by_group(
        self,
        etfs: List[str],
        prices: Dict[str, float],
        equity: float,
    ) -> float:
        """Fraction-of-equity notional for a subset of ETFs."""
        if equity <= 0:
            return 0.0
        total = sum(
            abs(self.etf_positions.get(e, 0.0)) * prices.get(e, 0.0)
            for e in etfs
        )
        return total / equity

    # ------------------------------------------------------------------
    # Order construction
    # ------------------------------------------------------------------

    def build_orders(
        self,
        mode:   str,
        equity: float,
        prices: Dict[str, float],
    ) -> List[ETFOrder]:
        """
        Build a list of ETF orders to transition toward *mode*.

        Parameters
        ----------
        mode   : ETFMode constant (HEDGE / AMPLIFY / FLAT).
        equity : Total portfolio equity in USD.
        prices : Current mid-prices for all ETF tickers.

        Returns
        -------
        List of ETFOrder objects (may be empty if market is closed,
        positions are already at target, or ETF layer is disabled).
        """
        if not self.enabled:
            return []

        if not is_etf_tradeable():
            period = get_market_period()
            print(f"[ETF LAYER] Market {period.value} — no ETF orders")
            return []

        order_type = get_etf_order_type()
        orders: List[ETFOrder] = []

        if mode == ETFMode.FLAT:
            orders = self._flatten_all(prices, equity, order_type)

        elif mode == ETFMode.HEDGE:
            # Flatten any long-amplifier positions first (mutual exclusivity)
            orders  = self._flatten_group(LONG_ETFS, prices, equity, order_type,
                                          reason="clearing longs before hedge")
            orders += self._build_hedge_orders(prices, equity, order_type)

        elif mode == ETFMode.AMPLIFY:
            # Flatten any short-hedge positions first (mutual exclusivity)
            orders  = self._flatten_group(SHORT_ETFS, prices, equity, order_type,
                                          reason="clearing hedges before amplify")
            orders += self._build_amplify_orders(prices, equity, order_type)

        return orders

    # ------------------------------------------------------------------
    # Internal order builders
    # ------------------------------------------------------------------

    def _build_hedge_orders(
        self,
        prices: Dict[str, float],
        equity: float,
        order_type: str,
    ) -> List[ETFOrder]:
        """
        Open / top-up short-ETF positions up to the hedge cap.

        Both SETH and ETHD target equal allocations, but only one needs
        to be active because they both track 3× inverse ETH — the bot
        can split the allocation or concentrate in one.
        Equal split is used here for diversification between issuers.
        """
        orders: List[ETFOrder] = []
        current_hedge_frac = self._notional_by_group(SHORT_ETFS, prices, equity)
        remaining_cap      = max(0.0, self.hedge_cap - current_hedge_frac)

        if remaining_cap < 0.001:   # < 0.1 % — not worth an order
            return orders

        # Split remaining cap equally between both short ETFs
        per_etf_frac = remaining_cap / len(SHORT_ETFS)

        for etf in SHORT_ETFS:
            price = prices.get(etf, 0.0)
            if price <= 0:
                print(f"[ETF LAYER] No price for {etf} — skipping hedge order")
                continue

            max_alloc  = ETF_DEFAULT_MAX_ALLOC.get(etf, 0.15)
            target_frac = min(per_etf_frac, max_alloc)
            current_frac = (
                abs(self.etf_positions.get(etf, 0.0)) * price / equity
                if equity > 0 else 0.0
            )
            delta_frac = target_frac - current_frac

            if delta_frac < 0.001:  # already at or above target
                continue

            coin_units  = (delta_frac * equity) / price
            limit_price = self._limit_price(price, "buy", order_type)

            orders.append(ETFOrder(
                asset       = etf,
                kraken_pair = ETF_KRAKEN_PAIRS[etf],
                side        = "buy",
                order_type  = order_type,
                coin_units  = round(coin_units, 8),
                limit_price = limit_price,
                reason      = f"hedge: adding {delta_frac:.2%} of equity via {etf}",
            ))

        return orders

    def _build_amplify_orders(
        self,
        prices: Dict[str, float],
        equity: float,
        order_type: str,
    ) -> List[ETFOrder]:
        """
        Open / top-up long-ETF positions up to the amplify cap.

        Each long ETF targets equal weight of the available cap.
        """
        orders: List[ETFOrder] = []
        current_amp_frac = self._notional_by_group(LONG_ETFS, prices, equity)
        remaining_cap    = max(0.0, self.amplify_cap - current_amp_frac)

        if remaining_cap < 0.001:
            return orders

        per_etf_frac = remaining_cap / len(LONG_ETFS)

        for etf in LONG_ETFS:
            price = prices.get(etf, 0.0)
            if price <= 0:
                print(f"[ETF LAYER] No price for {etf} — skipping amplify order")
                continue

            max_alloc   = ETF_DEFAULT_MAX_ALLOC.get(etf, 0.10)
            target_frac = min(per_etf_frac, max_alloc)
            current_frac = (
                abs(self.etf_positions.get(etf, 0.0)) * price / equity
                if equity > 0 else 0.0
            )
            delta_frac = target_frac - current_frac

            if delta_frac < 0.001:
                continue

            coin_units  = (delta_frac * equity) / price
            limit_price = self._limit_price(price, "buy", order_type)

            orders.append(ETFOrder(
                asset       = etf,
                kraken_pair = ETF_KRAKEN_PAIRS[etf],
                side        = "buy",
                order_type  = order_type,
                coin_units  = round(coin_units, 8),
                limit_price = limit_price,
                reason      = f"amplify: adding {delta_frac:.2%} of equity via {etf}",
            ))

        return orders

    def _flatten_group(
        self,
        etfs: List[str],
        prices: Dict[str, float],
        equity: float,
        order_type: str,
        reason: str = "flatten",
    ) -> List[ETFOrder]:
        """Sell entire position in each ETF in *etfs* to return to flat."""
        orders: List[ETFOrder] = []
        for etf in etfs:
            qty = self.etf_positions.get(etf, 0.0)
            if abs(qty) < 1e-8:
                continue
            price       = prices.get(etf, 0.0)
            if price <= 0:
                continue
            limit_price = self._limit_price(price, "sell", order_type)
            orders.append(ETFOrder(
                asset       = etf,
                kraken_pair = ETF_KRAKEN_PAIRS[etf],
                side        = "sell",
                order_type  = order_type,
                coin_units  = round(abs(qty), 8),
                limit_price = limit_price,
                reason      = reason,
            ))
        return orders

    def _flatten_all(
        self,
        prices: Dict[str, float],
        equity: float,
        order_type: str,
    ) -> List[ETFOrder]:
        """Flatten the entire ETF layer (neutral regime)."""
        return self._flatten_group(ALL_ETFS, prices, equity, order_type,
                                   reason="neutral regime — flatten ETF layer")

    # ------------------------------------------------------------------
    # Position update (called after fill confirmation)
    # ------------------------------------------------------------------

    def record_fill(self, asset: str, side: str, coin_units: float) -> None:
        """
        Update the internal position tracker after a confirmed fill.

        Parameters
        ----------
        asset      : ETF ticker (e.g. "SETH").
        side       : "buy" or "sell".
        coin_units : Unsigned fill quantity in base-asset units.
        """
        if asset not in self.etf_positions:
            self.etf_positions[asset] = 0.0
        delta = coin_units if side == "buy" else -coin_units
        self.etf_positions[asset] += delta
        print(
            f"[ETF LAYER] Fill: {side.upper()} {coin_units:.6f} {asset}  "
            f"pos={self.etf_positions[asset]:+.6f}"
        )

    # ------------------------------------------------------------------
    # Cap enforcement gate
    # ------------------------------------------------------------------

    def check_etf_cap(self, prices: Dict[str, float], equity: float) -> bool:
        """
        Return True if the current total ETF notional is within the cap.

        ``sum(|all_etf_positions × prices|) / equity ≤ total_cap``

        Call before executing new ETF orders to provide an independent
        safety check separate from the order-sizing logic above.
        """
        frac = self.current_etf_notional(prices, equity)
        ok   = frac <= self.total_cap + 1e-6   # small float tolerance
        if not ok:
            print(
                f"[ETF LAYER] CAP BREACH: current={frac:.2%} > cap={self.total_cap:.0%}"
                f" — order blocked"
            )
        return ok

    # ------------------------------------------------------------------
    # Limit price helper
    # ------------------------------------------------------------------

    def _limit_price(
        self,
        mid: float,
        side: str,
        order_type: str,
    ) -> Optional[float]:
        """
        Compute the limit price for an order, or None for market orders.

        Buys  → mid × (1 + tolerance)
        Sells → mid × (1 − tolerance)
        """
        if order_type == ORDER_TYPE_MARKET:
            return None
        tol = self.tolerance
        if side == "buy":
            return round(mid * (1.0 + tol), 8)
        return round(mid * (1.0 - tol), 8)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, prices: Dict[str, float], equity: float) -> dict:
        """Return a snapshot dict for logging and health checks."""
        return {
            "enabled":          self.enabled,
            "mode":             self._last_mode,
            "total_cap":        self.total_cap,
            "current_notional": self.current_etf_notional(prices, equity),
            "hedge_notional":   self._notional_by_group(SHORT_ETFS, prices, equity),
            "amplify_notional": self._notional_by_group(LONG_ETFS,  prices, equity),
            "positions":        dict(self.etf_positions),
        }
