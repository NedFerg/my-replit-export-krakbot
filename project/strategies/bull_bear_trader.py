"""
Bull/Bear Rotational Trading System
====================================
BullBearRotationalTrader orchestrates a 4-phase market-cycle trading strategy:

  "accumulation"   → Waiting for a confirmed BTC uptrend.  Zero trades.
  "bull_alt_season"→ BTC in sustained uptrend; alts rising. Spot + 2× longs.
  "alt_cascade"    → ETH reversed + legislation catalyst. Max commodity ETF.
  "bear_market"    → Market topping. 2× short ETFs, reduced/no spot exposure.

In addition to the 4-phase logic, a **hedge overlay** runs every bar
in every phase.  The overlay uses RSI, Bollinger Bands, and resistance /
support proximity to size a small bidirectional position:

  Short hedge (ETHD — 2× Short ETH ETP):
    Opened when overbought signals fire — RSI > 72, price above upper
    Bollinger Band, or price at rolling resistance.  Provides downside
    protection and profits if the market reverses.

  Long hedge floor (ETHU — 2× Long ETH ETP):
    Opened when oversold signals fire — RSI < 35, price below lower
    Bollinger Band, or price at rolling support.  Ensures the bot holds
    leveraged long exposure for a sharp bounce even if it hasn't yet
    transitioned to bull_alt_season.

Together these two legs mean the bot is positioned for the most probable
move but has resources on the other side so it can profit regardless of
whether the next move is up or down.

The strategy works from **any** BTC price level, not just above $100 K.
When BTC is at $74 K and trending upward the rolling-high breakout
component in BTCBreakoutDetector will accumulate confidence and trigger
bull_alt_season once the threshold is reached.

Configuration
-------------
All tuneable constants are at the top of this file.  The key parameters
are now readable from environment variables so no code changes are needed
to adapt to a different market entry price:

  BTC_ATH_TARGET           Hard price level for the absolute ATH bonus.
                            Set to 0 to rely entirely on rolling-high detection.
                            Default: 100000

  BTC_BULL_RUN_FLOOR       Minimum BTC price before the bot enters any bull
                            phase.  Guards against buying into a bear-market
                            dead-cat bounce.  Set this to just below the
                            current market price when starting the bot.
                            Default: 65000  (well below $74 K current price)

  BREAKOUT_CONFIDENCE_MIN  Confidence threshold to leave accumulation.
                            Default: 0.55  (lower than 0.60 to compensate for
                            the ATH component scoring 0 at $74 K)

  HEDGE_SHORT_MAX          Max short-ETF (ETHD) allocation as fraction of equity.
                            Default: 0.08 (8 %)

  HEDGE_LONG_MAX           Max long-ETF (ETHU) floor allocation as fraction of equity.
                            Default: 0.10 (10 %)

  HEDGE_RSI_OVERBOUGHT     RSI level that triggers the short hedge.
                            Default: 72

  HEDGE_RSI_OVERSOLD       RSI level that triggers the long hedge floor.
                            Default: 35

Integration
-----------
    from strategies.bull_bear_trader import BullBearRotationalTrader

    trader = BullBearRotationalTrader(broker=broker, archive=archive)
    # Call once per bar / loop tick:
    trader.step(prices)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from strategies.signals.btc_breakout_detector   import BTCBreakoutDetector
from strategies.signals.alt_pump_detector        import AltPumpDetector
from strategies.signals.market_topping_detector  import MarketToppingDetector
from strategies.signals.recovery_detector        import RecoveryDetector
from strategies.signals.hedge_signal_detector    import HedgeSignalDetector


# ---------------------------------------------------------------------------
# Configuration (mirrors problem spec)
# ---------------------------------------------------------------------------

COMMODITY_ETFS: list[str] = ["XXRP", "SLON", "ETHU"]
KEY_ALTS:       list[str] = ["SOL", "XRP", "HBAR", "LINK", "XLM"]
ALL_ASSETS:     list[str] = KEY_ALTS + COMMODITY_ETFS + ["ETHD", "SETH"]

# Mapping from leveraged/short ETF ticker → underlying spot asset.
# Used for price lookups: ETF prices are not available on the standard
# Kraken public ticker endpoint, so we use the underlying spot price for
# paper-fill calculations.
ETF_UNDERLYING: dict[str, str] = {
    "XXRP": "XRP",   # XRP 2× Long ETP
    "SLON": "SOL",   # SOL 2× Long ETP
    "ETHU": "ETH",   # ETH 2× Long ETP
    "ETHD": "ETH",   # ETH 2× Short ETP
    "SETH": "ETH",   # ETH 1× Short ETP
}

BTC_ATH_TARGET:         float = float(os.getenv("BTC_ATH_TARGET",          "100000"))
BREAKOUT_CONFIDENCE_MIN: float = float(os.getenv("BREAKOUT_CONFIDENCE_MIN",   "0.55"))
# Minimum BTC price before the bot enters any bull phase.
# Prevents buying into a dead-cat bounce during a true bear market.
# Set this below the current market price (e.g. $65 000 when BTC is at $74 K).
BTC_BULL_RUN_FLOOR:     float = float(os.getenv("BTC_BULL_RUN_FLOOR",        "65000"))

RSI_OVERBOUGHT:  int   = 80
RSI_UNDERBOUGHT: int   = 30
MIN_VOLUME_SPIKE: float = 1.5
CONSOLIDATION_DAYS: int = 5

MIN_ROTATION_GAIN: float = 0.10   # 10 % gain before considering rotation exit

# Position size bounds by phase (fraction of total portfolio)
SPOT_ALT_MIN:     float = 0.02
SPOT_ALT_MAX:     float = 0.15
LEVERAGE_ETF_MIN: float = 0.10
LEVERAGE_ETF_MAX: float = 0.20
BEAR_SHORT_MIN:   float = 0.15
BEAR_SHORT_MAX:   float = 0.25
CASCADE_ETF_MAX:  float = 0.30   # commodity ETF in alt_cascade phase

ROTATION_EXIT_FRACTION: float = 0.50   # exit 50 % of a peaked alt on rotation
ROTATION_ENTRY_SIZE:    float = 0.15   # enter 15 % in the next candidate

# ---------------------------------------------------------------------------
# Hedge overlay configuration
# ---------------------------------------------------------------------------
# The hedge overlay runs every bar regardless of the current phase.
# It reads RSI + Bollinger Bands + resistance/support proximity and sizes a
# small counter-position so the portfolio can profit from either direction.
#
# Short hedge (ETHD — 2× Short ETH ETP):
#   Opens when overbought signals fire (RSI > threshold, price above upper
#   Bollinger Band, or at rolling resistance).  Max size HEDGE_SHORT_MAX.
#
# Long hedge floor (ETHU — 2× Long ETH ETP):
#   Opens when oversold signals fire (RSI < threshold, price below lower
#   Bollinger Band, or at rolling support).  Max size HEDGE_LONG_MAX.
#
# The overlay is additive to the phase positions.  During bull_alt_season the
# long ETFs already carry most of the upside; the short hedge is a small
# insurance layer.  During accumulation both layers operate so the bot can
# capture a breakout in either direction.
#
# Override via environment variables:
HEDGE_SHORT_MAX: float = float(os.getenv("HEDGE_SHORT_MAX", "0.08"))   # 8 % cap
HEDGE_LONG_MAX:  float = float(os.getenv("HEDGE_LONG_MAX",  "0.10"))   # 10 % cap
HEDGE_RSI_OVERBOUGHT: int = int(os.getenv("HEDGE_RSI_OVERBOUGHT", "72"))
HEDGE_RSI_OVERSOLD:   int = int(os.getenv("HEDGE_RSI_OVERSOLD",   "35"))
# Assets the hedge overlay tracks (BTC drives the macro signal; ETH is the
# most liquid hedge vehicle).
HEDGE_ASSETS: list[str] = ["BTC", "ETH"]


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

PHASE_ACCUMULATION   = "accumulation"
PHASE_BULL_ALT       = "bull_alt_season"
PHASE_ALT_CASCADE    = "alt_cascade"
PHASE_BEAR_MARKET    = "bear_market"

VALID_PHASES = (PHASE_ACCUMULATION, PHASE_BULL_ALT, PHASE_ALT_CASCADE, PHASE_BEAR_MARKET)


class BullBearRotationalTrader:
    """
    Multi-phase rotational trading strategy for Kraken spot + leveraged ETFs.

    The trader is designed to be **signal-driven only** — it never places
    orders unless a quantitative signal justifies the trade.

    Parameters
    ----------
    broker  : A LiveBroker / PaperBroker instance used to execute orders.
              If None, the trader will compute signals but not execute.
    archive : A TradeArchive instance for persisting phase / rotation events.
              If None, no persistence is performed.
    initial_phase : Starting phase (default "accumulation").
    """

    def __init__(
        self,
        broker=None,
        archive=None,
        initial_phase: str = PHASE_ACCUMULATION,
    ) -> None:
        if initial_phase not in VALID_PHASES:
            raise ValueError(f"initial_phase must be one of {VALID_PHASES}")

        self.broker  = broker
        self.archive = archive

        # ---- Phase state machine ----------------------------------------
        self.phase: str = initial_phase
        self._phase_entered_at: float = time.time()

        # ---- Signal engines ---------------------------------------------
        self.btc_detector      = BTCBreakoutDetector(
            ath_target=BTC_ATH_TARGET,
            rolling_high_window=int(os.getenv("BTC_ROLLING_HIGH_WINDOW", "60")),
            rolling_high_hold=int(os.getenv("BTC_ROLLING_HIGH_HOLD",    "2")),
        )
        self.alt_detector      = AltPumpDetector(
            consolidation_bars=CONSOLIDATION_DAYS,
            volume_spike_ratio=MIN_VOLUME_SPIKE,
            rsi_overbought=RSI_OVERBOUGHT,
            rsi_underbought=RSI_UNDERBOUGHT,
        )
        self.topping_detector  = MarketToppingDetector(
            assets=KEY_ALTS,
            rsi_overbought=RSI_OVERBOUGHT,
            min_overbought=5,
        )
        self.recovery_detector = RecoveryDetector(assets=KEY_ALTS)

        # ---- Hedge overlay detectors ------------------------------------
        # One per tracked anchor asset (BTC + ETH).
        # Detectors are keyed by the spot asset name even though the actual
        # hedge instrument is an ETF (ETHD for short, ETHU for long).
        self.hedge_detectors: dict[str, HedgeSignalDetector] = {
            asset: HedgeSignalDetector(
                asset=asset,
                rsi_overbought=HEDGE_RSI_OVERBOUGHT,
                rsi_oversold=HEDGE_RSI_OVERSOLD,
                max_short=HEDGE_SHORT_MAX,
                max_long=HEDGE_LONG_MAX,
            )
            for asset in HEDGE_ASSETS
        }
        # Cache of last hedge recommendations (keyed by asset)
        self.hedge_recommendations: dict[str, Any] = {}

        # ---- Position tracking ------------------------------------------
        # Maps asset → fraction of portfolio currently allocated
        self.positions: dict[str, float] = {}
        # Maps asset → entry price for P&L tracking
        self._entry_prices: dict[str, float] = {}
        # Maps asset → timestamp of entry (for duration tracking)
        self._entry_times: dict[str, float] = {}

        # ---- Signal cache (updated each step) ---------------------------
        self.last_signals: dict[str, Any] = {}

        # ---- Log buffer (for display in main loop) ---------------------
        self._log_lines: list[str] = []

        print(
            f"[BullBearTrader] Initialized — phase: {self.phase}\n"
            f"  BTC_BULL_RUN_FLOOR      = ${BTC_BULL_RUN_FLOOR:,.0f}  "
            f"(bot enters bull phases only above this price)\n"
            f"  BTC_ATH_TARGET          = ${BTC_ATH_TARGET:,.0f}  "
            f"(absolute ATH bonus level)\n"
            f"  BREAKOUT_CONFIDENCE_MIN = {BREAKOUT_CONFIDENCE_MIN:.2f}  "
            f"(confidence threshold to exit accumulation)"
        )

    # ====================================================================
    # Main entry point
    # ====================================================================

    def step(self, prices: dict[str, float], volumes: dict[str, float] | None = None) -> None:
        """
        Called once per bar / loop tick.

        1. Updates all signal detectors with the latest prices.
        2. Evaluates phase-transition conditions.
        3. Executes phase-appropriate rotation logic.

        Parameters
        ----------
        prices  : Dict of {ticker: price}, e.g. {"BTC": 102000, "SOL": 200}.
        volumes : Optional dict of {ticker: volume} for the same bar.
        """
        volumes = volumes or {}
        self._log_lines.clear()

        # ---- Update signals --------------------------------------------
        btc_price = prices.get("BTC", 0.0)
        btc_vol   = volumes.get("BTC", 0.0)
        btc_confidence = self.btc_detector.update(btc_price, btc_vol)

        for asset in KEY_ALTS:
            price = prices.get(asset, 0.0)
            vol   = volumes.get(asset, 0.0)
            if price > 0:
                self.alt_detector.update(asset, price, vol)

        market_topping = self.topping_detector.update(prices)
        recovering     = self.recovery_detector.update(prices)

        # ---- Update hedge overlay detectors ----------------------------
        for asset in HEDGE_ASSETS:
            price = prices.get(asset, 0.0)
            vol   = volumes.get(asset, 0.0)
            if price > 0:
                rec = self.hedge_detectors[asset].update(price, vol)
                self.hedge_recommendations[asset] = rec

        self.last_signals = {
            "btc_price":            btc_price,
            "btc_confidence":       btc_confidence,
            "market_topping":       market_topping,
            "recovering":           recovering,
            "alt_scores":           dict(self.alt_detector.last_scores),
            "hedge_recommendations": dict(self.hedge_recommendations),
        }

        # ---- Phase transition logic ------------------------------------
        self._evaluate_phase_transition(prices, btc_confidence, market_topping, recovering)

        # ---- Phase execution logic ------------------------------------
        self._execute_phase(prices)

        # ---- Hedge overlay (runs every bar, every phase) ---------------
        self._apply_hedge_overlay(prices)

    # ====================================================================
    # Phase state machine
    # ====================================================================

    def _evaluate_phase_transition(
        self,
        prices: dict[str, float],
        btc_confidence: float,
        market_topping: bool,
        recovering: bool,
    ) -> None:
        """Check for phase transitions and execute them when triggered."""
        old_phase = self.phase

        if self.phase == PHASE_ACCUMULATION:
            btc_price = prices.get("BTC", 0.0)
            if btc_price >= BTC_BULL_RUN_FLOOR and btc_confidence >= BREAKOUT_CONFIDENCE_MIN:
                self._transition_to(
                    PHASE_BULL_ALT,
                    reason=f"BTC breakout confirmed (confidence={btc_confidence:.2f})",
                    prices=prices,
                    confidence=btc_confidence,
                )

        elif self.phase == PHASE_BULL_ALT:
            # Cascade trigger: ETH showing weakness AND/OR legislation catalyst
            eth_price = prices.get("ETH", 0.0)
            eth_topping = self.alt_detector.is_topping("ETH") if eth_price > 0 else False
            if eth_topping and btc_confidence >= 0.30:
                self._transition_to(
                    PHASE_ALT_CASCADE,
                    reason="ETH topping + BTC still elevated — alt cascade beginning",
                    prices=prices,
                    confidence=btc_confidence,
                )
            elif market_topping:
                self._transition_to(
                    PHASE_BEAR_MARKET,
                    reason="Market topping signals fired — entering bear phase",
                    prices=prices,
                    confidence=btc_confidence,
                )

        elif self.phase == PHASE_ALT_CASCADE:
            if market_topping:
                self._transition_to(
                    PHASE_BEAR_MARKET,
                    reason="Market-wide topping confirmed — rotating to shorts",
                    prices=prices,
                    confidence=btc_confidence,
                )

        elif self.phase == PHASE_BEAR_MARKET:
            if recovering:
                self._transition_to(
                    PHASE_ACCUMULATION,
                    reason="Recovery signals detected — entering accumulation",
                    prices=prices,
                    confidence=btc_confidence,
                )

        if self.phase != old_phase:
            self._log(f"[PHASE] {old_phase} → {self.phase}")

    def _transition_to(
        self,
        new_phase: str,
        reason: str,
        prices: dict[str, float],
        confidence: float,
    ) -> None:
        """Execute a phase transition, persisting it to the archive."""
        old_phase  = self.phase
        self.phase = new_phase
        self._phase_entered_at = time.time()

        ts = datetime.now(timezone.utc).isoformat()
        print(
            f"[BullBearTrader] PHASE TRANSITION: {old_phase} → {new_phase}\n"
            f"  Reason: {reason}\n"
            f"  BTC: ${prices.get('BTC', 0):,.0f}  Confidence: {confidence:.2f}"
        )

        if self.archive and hasattr(self.archive, "record_phase_transition"):
            self.archive.record_phase_transition(
                from_phase=old_phase,
                to_phase=new_phase,
                trigger_reason=reason,
                btc_price=prices.get("BTC", 0.0),
                signal_confidence=confidence,
            )

    # ====================================================================
    # Phase execution
    # ====================================================================

    def _execute_phase(self, prices: dict[str, float]) -> None:
        """Dispatch to the appropriate phase handler."""
        if self.phase == PHASE_ACCUMULATION:
            self._phase_accumulation()
        elif self.phase == PHASE_BULL_ALT:
            self._phase_bull_alt_season(prices)
        elif self.phase == PHASE_ALT_CASCADE:
            self._phase_alt_cascade(prices)
        elif self.phase == PHASE_BEAR_MARKET:
            self._phase_bear_market(prices)

    def _phase_accumulation(self) -> None:
        """Accumulation: stay in cash, no trades, wait for BTC breakout."""
        # No orders — signal-driven only; do nothing until breakout fires.
        pass

    def _phase_bull_alt_season(self, prices: dict[str, float]) -> None:
        """
        Bull alt season:
        - Hold / build spot positions in KEY_ALTS (2–15 % each)
        - Hold / build leveraged ETF positions (ETHU, SLON, XXRP) at 10–20 %
        - Rotate: exit peaked alts, enter next-best candidate
        """
        # --- Rotation check: exit alts that have topped -----------------
        for asset in list(self.positions.keys()):
            if asset not in KEY_ALTS:
                continue
            if self.alt_detector.is_topping(asset):
                entry = self._entry_prices.get(asset, prices.get(asset, 0.0))
                current_price = prices.get(asset, 0.0)
                gain = (current_price - entry) / max(entry, 1e-9) if entry > 0 else 0.0
                if gain >= MIN_ROTATION_GAIN:
                    self._rotate_out(asset, prices)

        # --- Entry: score remaining alts and enter best candidates ------
        scores = {
            a: self.alt_detector.last_scores.get(a, 0.0)
            for a in KEY_ALTS
            if a not in self.positions
        }
        for asset, score in sorted(scores.items(), key=lambda x: -x[1]):
            if score >= 0.50:   # meaningful signal threshold
                size = self._position_size_bull(score)
                self._open_position(asset, size, prices.get(asset, 0.0))

        # --- Leveraged ETFs: enter if not already held ------------------
        for etf, underlying in [("ETHU", "ETH"), ("SLON", "SOL"), ("XXRP", "XRP")]:
            if etf not in self.positions:
                underlying_score = self.alt_detector.last_scores.get(underlying, 0.0)
                if underlying_score >= 0.40:
                    size = self._position_size_bull_etf(underlying_score)
                    self._open_position(etf, size, prices.get(etf, prices.get(underlying, 0.0)))

    def _phase_alt_cascade(self, prices: dict[str, float]) -> None:
        """
        Alt cascade:
        - Max out commodity ETF positions (XXRP, SLON, ETHU)
        - Continue rotating between alts as they peak
        """
        for etf in COMMODITY_ETFS:
            current_size = self.positions.get(etf, 0.0)
            if current_size < CASCADE_ETF_MAX:
                ref_asset = {"XXRP": "XRP", "SLON": "SOL", "ETHU": "ETH"}.get(etf, etf)
                self._open_position(
                    etf,
                    CASCADE_ETF_MAX - current_size,
                    prices.get(etf, prices.get(ref_asset, 0.0)),
                )

        # Continue alt rotation
        self._phase_bull_alt_season(prices)

    def _phase_bear_market(self, prices: dict[str, float]) -> None:
        """
        Bear market:
        - Exit all spot longs (reduce to 0 % or 2 %)
        - Enter / hold 2× short ETF (ETHD): 15–25 %
        - No long leverage allowed
        """
        # Trim spot positions to near-zero
        for asset in list(self.positions.keys()):
            if asset in KEY_ALTS or asset in COMMODITY_ETFS:
                current = self.positions.get(asset, 0.0)
                if current > 0.02:
                    self._close_position(asset, current - 0.02, prices.get(asset, 0.0))

        # Enter short ETF if not held
        if "ETHD" not in self.positions:
            self._open_position("ETHD", BEAR_SHORT_MIN, prices.get("ETHD", prices.get("ETH", 0.0)))
        else:
            # Scale up to target if under-allocated
            current = self.positions.get("ETHD", 0.0)
            if current < BEAR_SHORT_MIN:
                self._open_position(
                    "ETHD",
                    BEAR_SHORT_MIN - current,
                    prices.get("ETHD", prices.get("ETH", 0.0)),
                )

    # ====================================================================
    # Position sizing
    # ====================================================================

    def _position_size_bull(self, score: float) -> float:
        """Scale spot alt size with signal confidence (2 %–15 %)."""
        return SPOT_ALT_MIN + (SPOT_ALT_MAX - SPOT_ALT_MIN) * min(score, 1.0)

    def _position_size_bull_etf(self, score: float) -> float:
        """Scale leveraged ETF size with underlying alt score (10 %–20 %)."""
        return LEVERAGE_ETF_MIN + (LEVERAGE_ETF_MAX - LEVERAGE_ETF_MIN) * min(score, 1.0)

    # ====================================================================
    # Hedge overlay
    # ====================================================================

    def _apply_hedge_overlay(self, prices: dict[str, float]) -> None:
        """
        Real-time bidirectional hedge overlay — runs every bar in every phase.

        Reads the hedge recommendations computed in step() for BTC and ETH
        and adjusts two instrument positions accordingly:

          ETHD (2× Short ETH ETP) — the short hedge leg
            Sized proportional to ``overbought_score``.  Opened when RSI is
            elevated, price is above the upper Bollinger Band, or price is
            testing rolling resistance.  Trimmed when overbought signals fade.

          ETHU (2× Long ETH ETP) — the long hedge floor
            Sized proportional to ``oversold_score``.  Opened when RSI is
            depressed, price is below the lower Bollinger Band, or price is
            testing rolling support.  Trimmed when oversold signals fade.

        The hedge sizes are the **maximum** of the BTC and ETH signals so
        that a strong signal on either anchor asset drives the overlay.

        In the bear_market phase the short hedge from ``_phase_bear_market``
        takes priority: this overlay will not reduce positions that the phase
        logic already sized to BEAR_SHORT_MIN or above.
        """
        if not self.hedge_recommendations:
            return

        eth_price = prices.get("ETH", 0.0)
        if eth_price <= 0:
            return

        # ---- Aggregate signals across anchor assets --------------------
        # Take the max signal strength so either BTC or ETH can trigger.
        ob_score = max(
            rec.overbought_score
            for rec in self.hedge_recommendations.values()
        )
        os_score = max(
            rec.oversold_score
            for rec in self.hedge_recommendations.values()
        )

        target_short = HEDGE_SHORT_MAX * ob_score   # target ETHD allocation
        target_long  = HEDGE_LONG_MAX  * os_score   # target ETHU allocation

        # ---- Short hedge: ETHD ----------------------------------------
        current_short = self.positions.get("ETHD", 0.0)

        if self.phase == PHASE_BEAR_MARKET:
            # Bear phase logic manages ETHD via _phase_bear_market().
            # The overlay only top-ups; it never reduces below BEAR_SHORT_MIN.
            target_short = max(target_short, BEAR_SHORT_MIN)

        if target_short > current_short + 0.005:
            # Need more short hedge
            delta = target_short - current_short
            self._open_position("ETHD", delta, eth_price)
            self._log(
                f"[HEDGE] Short overlay: ETHD +{delta*100:.1f}%  "
                f"(ob_score={ob_score:.2f})"
            )
        elif target_short < current_short - 0.005 and self.phase != PHASE_BEAR_MARKET:
            # Signal faded — trim short hedge (not in bear phase where it's deliberate)
            delta = current_short - target_short
            self._close_position("ETHD", delta, eth_price)
            self._log(
                f"[HEDGE] Short trimmed: ETHD -{delta*100:.1f}%  "
                f"(ob_score={ob_score:.2f})"
            )

        # ---- Long hedge floor: ETHU -----------------------------------
        current_long = self.positions.get("ETHU", 0.0)

        if self.phase == PHASE_BEAR_MARKET:
            # No long ETF positions in a bear market.
            if current_long > 0.005:
                self._close_position("ETHU", current_long, eth_price)
                self._log("[HEDGE] Bear phase: closing long ETF floor (ETHU)")
            return

        if target_long > current_long + 0.005:
            delta = target_long - current_long
            self._open_position("ETHU", delta, eth_price)
            self._log(
                f"[HEDGE] Long floor: ETHU +{delta*100:.1f}%  "
                f"(os_score={os_score:.2f})"
            )
        elif target_long < current_long - 0.005:
            delta = current_long - target_long
            self._close_position("ETHU", delta, eth_price)
            self._log(
                f"[HEDGE] Long trimmed: ETHU -{delta*100:.1f}%  "
                f"(os_score={os_score:.2f})"
            )

    # ====================================================================
    # Order helpers
    # ====================================================================

    def _open_position(self, asset: str, size: float, price: float) -> None:
        """
        Open or add to a position.

        Logs the intent and (when a broker is wired) submits the order.
        The position size is expressed as a fraction of total portfolio equity.
        """
        if size <= 0 or price <= 0:
            return

        existing = self.positions.get(asset, 0.0)
        self.positions[asset] = existing + size
        if asset not in self._entry_prices:
            self._entry_prices[asset] = price
        if asset not in self._entry_times:
            self._entry_times[asset] = time.time()

        self._log(f"[ORDER] BUY {asset}  +{size*100:.1f}% @ ${price:,.4f}  (phase={self.phase})")

        if self.broker is not None:
            self._submit_order(asset, "buy", size, price)

    def _close_position(self, asset: str, size: float, price: float) -> None:
        """Reduce or exit a position."""
        if size <= 0 or price <= 0:
            return

        current = self.positions.get(asset, 0.0)
        closed  = min(size, current)
        if closed <= 0:
            return

        new_pos = current - closed
        if new_pos <= 0.001:
            self.positions.pop(asset, None)
            self._entry_prices.pop(asset, None)
            self._entry_times.pop(asset, None)
        else:
            self.positions[asset] = new_pos

        entry = self._entry_prices.get(asset, price)
        gain  = (price - entry) / max(entry, 1e-9) * 100
        self._log(
            f"[ORDER] SELL {asset}  -{closed*100:.1f}% @ ${price:,.4f}  gain={gain:+.1f}%"
        )

        if self.broker is not None:
            self._submit_order(asset, "sell", closed, price)

    def _rotate_out(self, asset: str, prices: dict[str, float]) -> None:
        """
        Exit ROTATION_EXIT_FRACTION of a peaked alt and select the next best
        candidate for entry.
        """
        price   = prices.get(asset, 0.0)
        current = self.positions.get(asset, 0.0)
        exit_size = current * ROTATION_EXIT_FRACTION

        entry     = self._entry_prices.get(asset, price)
        gain_pct  = (price - entry) / max(entry, 1e-9) * 100 if entry > 0 else 0.0

        self._log(
            f"[ROTATE] Exiting {ROTATION_EXIT_FRACTION*100:.0f}% of {asset} "
            f"@ ${price:,.4f}  gain={gain_pct:+.1f}%"
        )
        self._close_position(asset, exit_size, price)

        # Find the best new candidate
        candidates = {
            a: self.alt_detector.last_scores.get(a, 0.0)
            for a in KEY_ALTS
            if a != asset and a not in self.positions
        }
        if not candidates:
            return

        next_asset = max(candidates.items(), key=lambda x: x[1])[0]
        next_score = candidates[next_asset]
        next_price = prices.get(next_asset, 0.0)

        self._log(
            f"[ROTATE] Entering {next_asset} @ ${next_price:,.4f}  "
            f"score={next_score:.2f}  size={ROTATION_ENTRY_SIZE*100:.0f}%"
        )
        self._open_position(next_asset, ROTATION_ENTRY_SIZE, next_price)

        if self.archive and hasattr(self.archive, "record_rotation"):
            self.archive.record_rotation(
                from_asset=asset,
                to_asset=next_asset,
                from_exit_price=price,
                from_position_pct=exit_size,
                to_entry_price=next_price,
                to_position_pct=ROTATION_ENTRY_SIZE,
                from_gain_pct=gain_pct,
                rationale=f"Topping signal on {asset}; pump score {next_score:.2f} on {next_asset}",
            )

    def _submit_order(self, asset: str, side: str, size: float, price: float) -> None:
        """
        Route an order through the broker for paper/live execution.

        For PaperBroker (sandbox mode): calls broker.execute_trade() which
        routes to _paper_fill() — producing a synthetic fill with realistic
        slippage + fees, updating cash/positions, and logging to CSV/SQLite.

        For LiveBroker: same execute_trade() path, which places a real limit
        order on Kraken (guarded by dry_run / kill-switch).

        Parameters
        ----------
        asset : Internal asset ticker (e.g. "SOL", "ETHU").
        side  : "buy" or "sell".
        size  : Fractional portfolio allocation (0.10 = 10% of equity).
        price : Reference price used to convert the fraction to coin units.
                For ETF tickers (ETHU, ETHD, etc.) this is the underlying
                spot price; the broker will use self.live_prices for the fill.
        """
        # For ETF tokens, resolve the tradeable underlying asset and its price
        # since the broker's live_prices only has spot data.
        trade_asset = ETF_UNDERLYING.get(asset, asset)
        trade_price = price

        # Derive the equity base for converting fractional size → coin units
        if hasattr(self.broker, "compute_total_equity"):
            equity = self.broker.compute_total_equity() or 10_000.0
        else:
            equity = 10_000.0   # fallback: assume $10 K starting capital

        if equity <= 0 or trade_price <= 0:
            return

        coin_units = (size * equity) / trade_price
        if coin_units <= 0:
            return

        if hasattr(self.broker, "execute_trade"):
            self.broker.execute_trade(trade_asset, side, coin_units)
        else:
            self._log(f"[WARN] Broker has no execute_trade method — order for {asset} skipped")

    # ====================================================================
    # Reporting / display helpers
    # ====================================================================

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        self._log_lines.append(line)
        print(f"[BullBearTrader] {line}")

    def status_summary(self) -> dict[str, Any]:
        """Return a dict suitable for display in the main loop."""
        return {
            "phase":          self.phase,
            "positions":      dict(self.positions),
            "btc_confidence": self.last_signals.get("btc_confidence", 0.0),
            "market_topping": self.last_signals.get("market_topping", False),
            "recovering":     self.last_signals.get("recovering", False),
            "alt_scores":     self.last_signals.get("alt_scores", {}),
        }

    def print_status(self) -> None:
        """Print a formatted status block to stdout."""
        s = self.status_summary()
        print(
            f"\n{'='*60}\n"
            f"[BullBearTrader] Phase: {s['phase'].upper()}\n"
            f"  BTC breakout confidence : {s['btc_confidence']:.2f}\n"
            f"  Market topping          : {s['market_topping']}\n"
            f"  Recovery signal         : {s['recovering']}\n"
            f"  Current positions       : {s['positions']}\n"
            f"  Alt scores              : {s['alt_scores']}\n"
            f"{'='*60}\n"
        )
