"""
project/broker/etf_hedging.py
------------------------------
ETF hedging strategy and order builder.

This module replaces the disabled futures hedging layer for US users by
using Kraken Spot positions in crypto ETPs (bare ticker symbols, no suffix):

  Long ETPs:
    ETHU  — ETH 2× Long ETP
    SLON  — SOL 2× Long ETP
    XXRP  — XRP 2× Long ETP

  Short ETPs (only these two are short):
    ETHD  — ETH 2× Short ETP
    SETH  — ETH 1× Short ETP

Because Ethereum leads the altcoin market, shorting SETH/ETHD acts as a broad
crypto hedge during bearish trends — the same role futures hedging played.

Design
------
• ETF positions are stored separately in broker.etf_positions (never mixed
  with spot crypto positions in broker.spot_positions).
• Combined |ETHD| + |SETH| notional ≤ 30 % of total equity at all times.
• Order type (market vs. limit) is enforced by MarketHours:
    – Regular hours (09:30–16:00 ET): market orders
    – Pre/after-market (04:00–09:30 and 16:00–20:00 ET): limit orders only
    – Closed (overnight / weekends): no ETF orders

The module is intentionally thin: it computes *what* to do and returns an
order dict.  The broker decides *whether* to submit it (kill-switch checks,
daily-loss caps, etc. still apply in the broker layer).

Usage (called from broker.run_etf_overlay())
-----
    from broker.etf_hedging import ETFHedger

    hedger = ETFHedger(max_etf_allocation=0.30)
    orders = hedger.compute_orders(
        regime       = {"cycle_phase": 3, "panic_risk": 2},
        equity       = 10_000.0,
        etf_prices   = {"ETHD": 25.0, "SETH": 8.0},
        etf_positions= {"ETHD": 0.0, "SETH": 0.0},
    )
    # orders is a list of dicts with keys: asset, side, units, order_type, price
"""

import enum
import os
from typing import NamedTuple

from utils.market_hours import MarketHours, MarketSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ETF_ASSETS = ("ETHU", "SLON", "XXRP", "ETHD", "SETH")

# Alias used by broker.py for ETF position initialisation.
ALL_ETFS = ETF_ASSETS

# Default Kraken trading pairs for each ETF ticker.
# ETFs/ETPs on Kraken use bare ticker symbols (no currency suffix).
ETF_KRAKEN_PAIRS: dict = {
    "ETHU": "ETHU",      # ETH 2× Long ETP
    "SLON": "SLON",      # SOL 2× Long ETP
    "XXRP": "XXRP",      # XRP 2× Long ETP
    "ETHD": "ETHD",      # ETH 2× Short ETP
    "SETH": "SETH",      # ETH 1× Short ETP
}


class ETFMode(str, enum.Enum):
    """Operating mode for the ETF hedging layer."""
    DISABLED = "disabled"
    HEDGE    = "hedge"
    AMPLIFY  = "amplify"


class ETFOrder(NamedTuple):
    """
    Typed container for a single ETF order produced by ETFHedger.

    Fields
    ------
    asset      : ticker symbol, e.g. "ETHD" or "SETH"
    side       : "buy" or "sell"
    units      : unsigned coin quantity to trade
    order_type : "market" or "limit"
    price      : limit price in the quote currency; 0.0 for market orders
    notional   : estimated order value in the quote currency (|units × price|)
    """
    asset:      str
    side:       str
    units:      float
    order_type: str
    price:      float
    notional:   float

# Default allocation cap — overridden by MAX_ETF_ALLOCATION env var or the
# max_etf_allocation constructor parameter.
DEFAULT_MAX_ETF_ALLOCATION = float(os.getenv("MAX_ETF_ALLOCATION", "0.30"))

# Minimum notional USD per order — skip dust trades
MIN_ORDER_USD = float(os.getenv("ETF_MIN_ORDER_USD", "5.0"))

# Limit order tolerance: how far from mid the limit price is placed.
# 0.1 % matches the spot limit_order_tolerance default in broker.py.
DEFAULT_LIMIT_TOLERANCE = float(os.getenv("ETF_LIMIT_TOLERANCE", "0.001"))


# ---------------------------------------------------------------------------
# Regime → target-exposure mapping
# ---------------------------------------------------------------------------

def _regime_to_etf_targets(regime: dict) -> dict:
    """
    Map a market regime snapshot to target fractional ETF exposures.

    Inputs (all optional — missing keys default to neutral):
      cycle_phase  int 0-3   (0=accumulation, 1=expansion, 2=distribution, 3=reset)
      panic_risk   int 0-2   (0=none, 1=moderate, 2=severe)
      top_risk     int 0-2   (0=none, 1=moderate, 2=severe)
      vol_scaler   float     (>1 = elevated volatility)

    Target exposure conventions:
      ETHD  ≥ 0  (long exposure; 0 = no position)
      SETH  ≥ 0  (short exposure expressed as a positive allocation fraction;
                  the order side is always "buy" when opening, "sell" when closing)

    Returns dict {"ETHD": float, "SETH": float} — each in [0, 1].
    The caller enforces the 30 % portfolio cap after combining both.
    """
    cycle_phase = int(regime.get("cycle_phase", 1))
    panic_risk  = int(regime.get("panic_risk",  0))
    top_risk    = int(regime.get("top_risk",    0))
    vol_scaler  = float(regime.get("vol_scaler", 1.0))

    # Severe panic → max short hedge; minimal or no long
    if panic_risk == 2:
        return {"ETHD": 0.00, "SETH": 0.30}

    # Severe blow-off top → reduce long, open moderate short
    if top_risk == 2:
        return {"ETHD": 0.00, "SETH": 0.20}

    # Moderate panic → partial short, no long
    if panic_risk == 1:
        return {"ETHD": 0.00, "SETH": 0.15}

    # Moderate top → trim long, small short
    if top_risk == 1:
        return {"ETHD": 0.05, "SETH": 0.10}

    # Cycle-phase base targets (no extreme signals)
    phase_map = {
        0: {"ETHD": 0.10, "SETH": 0.00},   # accumulation: small long hedge
        1: {"ETHD": 0.15, "SETH": 0.00},   # expansion: moderate long hedge
        2: {"ETHD": 0.05, "SETH": 0.05},   # distribution: balanced
        3: {"ETHD": 0.00, "SETH": 0.20},   # reset/crash: short hedge
    }
    targets = phase_map.get(cycle_phase, {"ETHD": 0.0, "SETH": 0.0})

    # Scale down slightly when volatility is very elevated
    if vol_scaler > 1.5:
        scale = max(0.5, 1.0 / vol_scaler)
        targets = {k: v * scale for k, v in targets.items()}

    return targets


# ---------------------------------------------------------------------------
# Order builder
# ---------------------------------------------------------------------------

class ETFHedger:
    """
    Stateless ETF hedging strategy.

    All state (current positions, prices) is passed in by the caller so this
    object has no internal mutable state and is safe to call from any thread.
    """

    def __init__(
        self,
        max_etf_allocation: float = DEFAULT_MAX_ETF_ALLOCATION,
        limit_tolerance: float    = DEFAULT_LIMIT_TOLERANCE,
    ):
        self.max_etf_allocation = max_etf_allocation
        self.limit_tolerance    = limit_tolerance
        self._market_hours      = MarketHours()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_orders(
        self,
        regime:        dict,
        equity:        float,
        etf_prices:    dict,
        etf_positions: dict,
        now=None,
    ) -> list:
        """
        Compute ETF rebalance orders for the current regime and positions.

        Parameters
        ----------
        regime        : dict with keys cycle_phase, panic_risk, top_risk, vol_scaler
        equity        : total portfolio equity in USD
        etf_prices    : {"ETHD": float, "SETH": float} — current mid-prices
        etf_positions : {"ETHD": float, "SETH": float} — current coin holdings
        now           : optional datetime for market-hours determination (for tests)

        Returns
        -------
        list of order dicts, each with:
            {
                "asset":      str,    # "ETHD" or "SETH"
                "side":       str,    # "buy" or "sell"
                "units":      float,  # unsigned coin quantity
                "order_type": str,    # "market" or "limit"
                "price":      float,  # limit price (0.0 when market order)
                "notional":   float,  # estimated USD notional
            }

        An empty list means no action is required.
        """
        if equity <= 0:
            return []

        if not self._market_hours.etf_trading_allowed(now):
            return []

        session    = self._market_hours.get_session(now)
        order_type = self._market_hours.required_order_type(now)

        targets = _regime_to_etf_targets(regime)

        # Enforce combined 30 % cap
        total_target = sum(targets.values())
        if total_target > self.max_etf_allocation:
            scale   = self.max_etf_allocation / total_target
            targets = {k: v * scale for k, v in targets.items()}

        orders = []
        for asset in ETF_ASSETS:
            price = etf_prices.get(asset, 0.0)
            if price <= 0:
                continue

            target_frac   = targets.get(asset, 0.0)
            target_usd    = target_frac * equity
            target_units  = target_usd / price

            current_units = float(etf_positions.get(asset, 0.0))
            current_usd   = current_units * price

            delta_usd   = target_usd - current_usd
            delta_units = abs(delta_usd) / price

            if abs(delta_usd) < MIN_ORDER_USD:
                continue

            side = "buy" if delta_usd > 0 else "sell"

            if order_type == "limit":
                if side == "buy":
                    limit_price = round(price * (1.0 + self.limit_tolerance), 8)
                else:
                    limit_price = round(price * (1.0 - self.limit_tolerance), 8)
            else:
                limit_price = 0.0   # market order — no explicit price needed

            orders.append({
                "asset":      asset,
                "side":       side,
                "units":      round(delta_units, 8),
                "order_type": order_type,
                "price":      limit_price,
                "notional":   round(abs(delta_usd), 4),
            })

            print(
                f"[ETF HEDGER] {asset}  session={session}"
                f"  target={target_frac:.3f}  current={current_usd/equity:.3f}"
                f"  delta_usd={delta_usd:+.2f}  side={side}"
                f"  order_type={order_type}"
                + (f"  limit_price={limit_price:.4f}" if order_type == "limit" else "")
            )

        return orders

    def etf_portfolio_fraction(
        self,
        equity:        float,
        etf_positions: dict,
        etf_prices:    dict,
    ) -> float:
        """
        Return the current combined ETF allocation as a fraction of equity.

        sum(|ETHD_usd| + |SETH_usd|) / equity
        """
        if equity <= 0:
            return 0.0
        total = sum(
            abs(etf_positions.get(a, 0.0)) * etf_prices.get(a, 0.0)
            for a in ETF_ASSETS
        )
        return total / equity

    def cap_breached(
        self,
        equity:        float,
        etf_positions: dict,
        etf_prices:    dict,
    ) -> bool:
        """Return True if the current ETF allocation exceeds the 30 % cap."""
        return self.etf_portfolio_fraction(equity, etf_positions, etf_prices) > self.max_etf_allocation


# ---------------------------------------------------------------------------
# Alias — broker.py imports ETFHedgingLayer; ETFHedger is the implementation.
# ---------------------------------------------------------------------------

ETFHedgingLayer = ETFHedger
