import collections
import math
import random

import numpy as np

# ---------------------------------------------------------------------------
# Macro feedback thresholds (tune these to match your vol/alt scale)
# ---------------------------------------------------------------------------
VOL_HIGH_THRESHOLD = 0.02   # vol_regime above this is "stressed"
VOL_LOW_THRESHOLD  = 0.005  # vol_regime below this is "calm"
ALT_HIGH_THRESHOLD = 0.6    # altseason_index above this is "risk-on"
ALT_LOW_THRESHOLD  = 0.2    # altseason_index below this is "risk-off"

from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.rl_agent import ReinforcementLearningTrader
from agents.market_agent import MarketAgent
from broker.broker import SimulatedBroker
from market_data.data_source import SimulatedDataSource
from risk.risk_manager import RiskManager, OrderIntent
from config.config import (
    SIMULATION_STEPS,
    INITIAL_BALANCE,
    MARKET_START_PRICE
)


class Simulation:
    """
    Orchestrates one episode of the multi-agent market simulation.

    All market data flows in through a MarketDataSource (Tick objects).
    All order execution flows out through a Broker, which returns Fill objects.

    Latency model
    -------------
    When an agent decides on an action, the resulting OrderIntent is not sent
    to the broker immediately.  Instead it is placed on a latency_queue with a
    delivery step of  current_step + agent.latency.  At the start of each step
    the queue is drained: any intent whose delivery step has been reached is
    risk-checked and forwarded to the broker using the *current* market state.
    This means a delayed order may be rejected (or fill at a stale price) if the
    market has moved before the order arrives — realistic behaviour for slow or
    high-latency agents.

    Fee model
    ---------
    Fees are applied inside the Exchange during matching and are deducted
    directly from the agent's balance.  Because equity = balance + unrealized_pnl,
    fees naturally reduce equity and flow into the RL reward signal without any
    extra reward shaping.
    """

    def __init__(self, agents=None, market_data_source=None, broker=None, risk_manager=None):
        # --- Market data source ------------------------------------------
        if market_data_source is not None:
            self.market_data_source = market_data_source
        else:
            self.market_data_source = SimulatedDataSource(
                MarketAgent(MARKET_START_PRICE)
            )

        self.initial_regime = self.market_data_source.initial_regime

        # --- Broker (order execution) ------------------------------------
        self.broker = broker if broker is not None else SimulatedBroker()

        # --- Agents -------------------------------------------------------
        if agents is not None:
            self.agents = agents
        else:
            self.agents = [
                ValueTrader("ValueTrader", INITIAL_BALANCE),
                MomentumTrader("MomentumTrader", INITIAL_BALANCE),
                RandomTrader("RandomTrader", INITIAL_BALANCE),
                ReinforcementLearningTrader("RLTrader", INITIAL_BALANCE),
            ]

        # --- Risk manager ------------------------------------------------
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager()

        # --- Latency queue: (deliver_step, agent, order_intent) ----------
        self.latency_queue = []

        # --- Histories ---------------------------------------------------
        self.price_history = []
        self.regime_history = []

        # --- Rolling volatility tracking ----------------------------------
        # 1-step log-returns are stored; vols are computed from these each step.
        self.short_vol_window = 5
        self.long_vol_window = 20
        self.return_history = collections.deque(maxlen=self.long_vol_window)
        self._prev_price = None   # tracks last price for return computation

        # --- Risk-aware reward: RL-agent equity tracking -----------------
        # equity_peak and equity_history are reset each episode because a
        # fresh Simulation is constructed per episode in main.py.
        self.equity_peak = None
        self.equity_history = []

        # --- Multi-timeframe trend windows --------------------------------
        self.trend_windows = [5, 20, 50]
        # price_history (defined in __init__ above) stores raw prices per step
        # and is reused here for momentum lookback.

        # --- Volume and order-flow tracking ------------------------------
        self.volume_history = []
        self.buy_volume_history = []
        self.sell_volume_history = []
        self.volume_window = 20

        # --- Microstructure parameters (tunable) -------------------------
        # All values in basis points (1 bp = 0.01%).
        self.spread_bps  = 10   # 0.10% half-spread each side
        self.txn_cost_bps = 2   # 0.02% flat fee per trade

        # --- Market impact parameters (tunable) --------------------------
        # Dynamic slippage depends on volatility, order-flow pressure,
        # trade size, and a liquidity proxy (rolling volume).
        self.impact_vol_coeff  = 0.5   # slippage grows with equity volatility
        self.impact_flow_coeff = 0.3   # slippage grows with order-flow imbalance
        self.impact_size_coeff = 1.0   # slippage grows with exposure-change size
        self.min_liquidity     = 1e-6  # floor to avoid division-by-zero

        # Populated in run() so apply_microstructure() can read it.
        self.current_state = None

        # --- Multi-asset universe ------------------------------------------
        # 8 assets: SOL is the primary agent-traded asset; the rest are
        # traded by the RL portfolio and also feed cross-asset state features.
        # BTC and ETH are anchor/benchmark assets for the rotation engine.
        self.assets = ["SOL", "XRP", "LINK", "ETH", "BTC", "XLM", "HBAR", "AVAX"]

        self.asset_prices             = {a: [] for a in self.assets}
        self.asset_returns            = {a: [] for a in self.assets}
        self.asset_vol                = {a: 0.0 for a in self.assets}
        self.asset_mom_5              = {a: 0.0 for a in self.assets}
        self.asset_mom_20             = {a: 0.0 for a in self.assets}
        self.asset_mom_50             = {a: 0.0 for a in self.assets}
        self.asset_volume_history     = {a: [] for a in self.assets}
        self.asset_buy_volume_history = {a: [] for a in self.assets}
        self.asset_sell_volume_history= {a: [] for a in self.assets}
        self.asset_rolling_vol        = {a: 0.0 for a in self.assets}
        self.asset_vol_imbalance      = {a: 0.0 for a in self.assets}
        self.asset_pressure           = {a: 0.0 for a in self.assets}

        # --- Crypto-regime feature buffers --------------------------------
        self.btc_dominance    = 0.0   # BTC market-cap share of total
        self.eth_btc_ratio    = 0.0   # ETH price / BTC price
        self.sol_btc_strength = 0.0   # SOL price / BTC price
        self.sol_eth_strength = 0.0   # SOL price / ETH price
        self.altseason_index  = 0.0   # composite altcoin momentum signal
        self.vol_regime       = 0.0   # cross-market realized volatility
        self.liq_regime       = 0.0   # cross-market average rolling volume

        # --- Macro regime flag (set externally or via config) -------------
        # -1 = bear, 0 = neutral, +1 = bull, +2 = AUTO (hands-off)
        # AUTO: system ignores the manual override and derives its own
        #       internal regime from market signals each step.
        self.macro_regime    = 0
        self.internal_regime = 0   # computed each step in AUTO mode

        # --- Dynamic exposure caps (updated every step) -------------------
        self.dynamic_caps      = {}   # asset → current cap
        self.prev_dynamic_caps = {}   # asset → cap from previous step

        # --- Blow-off top risk flag ----------------------------------------
        # 0 = normal, 1 = early warning, 2 = blow-off (de-risk aggressively)
        self.top_risk      = 0
        self.prev_top_risk = 0

        # --- Multi-asset rotation engine ----------------------------------
        # rotation_score[asset] ∈ [0, ~4.5] — higher = current leader
        self.rotation_score      = {}   # asset → score this step
        self.prev_rotation_score = {}   # asset → score last step (for feedback)

        # --- Macro feedback message log -----------------------------------
        # Populated each step when manual flag conflicts with signals,
        # or when AUTO mode detects uncertain / mixed conditions.
        # Console print is throttled to only fire on *transitions*
        # (when the message changes) to avoid per-step noise.
        self.macro_messages      = []
        self._last_macro_message = None   # dedup: only print on change

    # ------------------------------------------------------------------
    # Microstructure helper
    # ------------------------------------------------------------------

    def apply_microstructure(self, mid_price, side, delta_exposure):
        """
        Compute the realistic execution price and flat transaction cost
        for a single RL-agent exposure adjustment.

        Slippage is now dynamic and state-dependent:
          - grows with current equity volatility
          - grows with order-flow pressure (bid/sell imbalance)
          - grows with the size of the exposure change
          - grows when rolling volume (liquidity proxy) is thin

        Parameters
        ----------
        mid_price      : float  — current mid market price
        side           : 'buy' or 'sell'
        delta_exposure : float  — signed exposure change this step

        Returns
        -------
        exec_price : float — price at which the trade executes
        txn_cost   : float — flat fee charged regardless of direction
        """
        state = self.current_state

        # --- Static components -----------------------------------------
        spread   = mid_price * (self.spread_bps  / 10_000)
        txn_cost = mid_price * (self.txn_cost_bps / 10_000)

        # --- Dynamic market impact slippage ----------------------------
        # Volatility-based impact: higher realized vol → wider impact.
        # Uses equity_vol from the previous step (lagged by one step because
        # execute_exposure runs before the tracking section updates it).
        equity_vol = getattr(state, "equity_vol", 0.0) if state else 0.0
        vol_impact = equity_vol * self.impact_vol_coeff

        # Order-flow imbalance impact: strong directional flow costs more.
        pressure    = getattr(state, "pressure",    0.0) if state else 0.0
        flow_impact = abs(pressure) * self.impact_flow_coeff

        # Trade-size impact: larger exposure changes move the market more.
        size_impact = abs(delta_exposure) * self.impact_size_coeff

        # Liquidity adjustment: thin markets amplify all components.
        # rolling_vol is the 20-bar average synthetic volume (raw, un-normalised).
        rolling_vol = getattr(state, "rolling_vol", 0.0) if state else 0.0
        liquidity   = max(rolling_vol, self.min_liquidity)
        liq_factor  = 1.0 / liquidity

        # Total impact (capped at 2% to prevent blow-up in the first 20 bars
        # before rolling_vol has accumulated and liq_factor is still very large).
        impact  = (vol_impact + flow_impact + size_impact) * liq_factor
        impact  = min(impact, 0.02)

        slippage = mid_price * impact

        if side == "buy":
            exec_price = mid_price + spread / 2 + slippage
        else:
            exec_price = mid_price - spread / 2 - slippage

        return exec_price, txn_cost

    def run(self):
        # Register agents: records starting equity, clears per-episode counters
        self.risk_manager.register_agents(self.agents)

        for current_step in range(SIMULATION_STEPS):
            # --- Consume one tick from the data source -------------------
            tick = self.market_data_source.get_next_tick()
            price = tick.price

            # --- Build MarketState via the broker ------------------------
            state = self.broker.get_market_state(
                price,
                tick.regime,
                tick.drift,
                tick.volatility,
            )

            # --- Rolling realized volatility -----------------------------
            # Compute 1-step return and update the rolling return history.
            # Then attach short_vol and long_vol as extra state attributes
            # so agents can consume them without changing the broker or
            # MarketState class.
            if self._prev_price is not None and self._prev_price > 0:
                price_return = (price - self._prev_price) / self._prev_price
            else:
                price_return = 0.0
            self.return_history.append(price_return)
            self._prev_price = price

            if len(self.return_history) >= self.short_vol_window:
                short_slice = list(self.return_history)[-self.short_vol_window:]
                short_mean = sum(short_slice) / len(short_slice)
                short_var = sum((r - short_mean) ** 2 for r in short_slice) / max(len(short_slice) - 1, 1)
                state.short_vol = math.sqrt(short_var)
            else:
                state.short_vol = 0.0

            if len(self.return_history) >= self.long_vol_window:
                long_slice = list(self.return_history)
                long_mean = sum(long_slice) / len(long_slice)
                long_var = sum((r - long_mean) ** 2 for r in long_slice) / max(len(long_slice) - 1, 1)
                state.long_vol = math.sqrt(long_var)
            else:
                state.long_vol = 0.0

            # --- Multi-timeframe momentum ---------------------------------
            # price_history does NOT yet contain the current price (it is
            # appended at the end of the loop), so self.price_history[-w]
            # is exactly the price w steps ago — no off-by-one adjustment.
            for w in self.trend_windows:
                if len(self.price_history) >= w:
                    past_price = self.price_history[-w]
                    mom = (price - past_price) / past_price if past_price != 0 else 0.0
                else:
                    mom = 0.0
                setattr(state, f"mom_{w}", mom)

            # --- Volume and order-flow simulation -------------------------
            # prev_price: use the last entry in price_history (current price
            # is not appended until end of loop, so [-1] is one step back).
            _prev_p = self.price_history[-1] if self.price_history else price
            _price_move = abs(price - _prev_p)

            # Base volume proportional to absolute price move
            # (more volatility → more microstructure activity).
            base_vol = max(1.0, _price_move * 1000)
            volume   = base_vol * random.uniform(0.8, 1.2)

            # Split directionally: whichever side drove the move gets
            # a majority share (55–75%); the other side absorbs the rest.
            if price > _prev_p:
                buy_vol  = volume * random.uniform(0.55, 0.75)
                sell_vol = volume - buy_vol
            elif price < _prev_p:
                sell_vol = volume * random.uniform(0.55, 0.75)
                buy_vol  = volume - sell_vol
            else:
                buy_vol  = volume * 0.5
                sell_vol = volume * 0.5

            self.volume_history.append(volume)
            self.buy_volume_history.append(buy_vol)
            self.sell_volume_history.append(sell_vol)

            # Rolling average volume over the window
            if len(self.volume_history) >= self.volume_window:
                vol_slice   = self.volume_history[-self.volume_window:]
                rolling_vol = sum(vol_slice) / len(vol_slice)
            else:
                rolling_vol = 0.0

            # Imbalance: signed difference between buy and sell pressure
            vol_imbalance = buy_vol - sell_vol

            # Pressure: imbalance normalised to [−1, +1]
            _denom   = buy_vol + sell_vol
            pressure = vol_imbalance / _denom if _denom > 0 else 0.0

            state.rolling_vol   = rolling_vol
            state.vol_imbalance = vol_imbalance
            state.pressure      = pressure

            # --- Multi-asset price updates --------------------------------
            # SOL: map current tick price into asset_prices so the per-asset
            # feature loops below can treat all assets uniformly.
            self.asset_prices["SOL"].append(price)

            # Non-SOL assets: generate correlated synthetic prices.
            # Ordering matters: BTC first → ETH follows BTC → LINK, XRP,
            # XLM, HBAR, AVAX follow ETH.  Same-step references are always
            # populated before they are read.
            for a in ["BTC", "ETH", "LINK", "XRP", "XLM", "HBAR", "AVAX"]:
                prev_a = self.asset_prices[a][-1] if self.asset_prices[a] else 100.0
                if a == "BTC":
                    a_drift = random.uniform(-0.002, 0.002)
                elif a == "ETH":
                    a_drift = (random.uniform(-0.003, 0.003)
                               + 0.3 * (self.asset_prices["BTC"][-1] - prev_a) / prev_a)
                else:  # LINK, XRP follow ETH
                    a_drift = (random.uniform(-0.004, 0.004)
                               + 0.2 * (self.asset_prices["ETH"][-1] - prev_a) / prev_a)
                self.asset_prices[a].append(max(0.1, prev_a * (1 + a_drift)))

            # --- Per-asset returns, volatility and momentum ---------------
            for a in self.assets:
                a_prices = self.asset_prices[a]
                if len(a_prices) < 2:
                    continue
                a_ret = (a_prices[-1] - a_prices[-2]) / a_prices[-2]
                self.asset_returns[a].append(a_ret)

                a_rets = self.asset_returns[a]
                self.asset_vol[a]    = float(np.std(a_rets[-20:])) if len(a_rets) >= 20 else 0.0
                self.asset_mom_5[a]  = a_prices[-1] - a_prices[-5]  if len(a_prices) >= 5  else 0.0
                self.asset_mom_20[a] = a_prices[-1] - a_prices[-20] if len(a_prices) >= 20 else 0.0
                self.asset_mom_50[a] = a_prices[-1] - a_prices[-50] if len(a_prices) >= 50 else 0.0

            # --- Per-asset synthetic volume and order-flow features -------
            for a in self.assets:
                if len(self.asset_prices[a]) < 2:
                    continue
                prev_ap = self.asset_prices[a][-2]
                curr_ap = self.asset_prices[a][-1]

                base_vol_a = max(1.0, abs(curr_ap - prev_ap) * 1000)
                volume_a   = base_vol_a * random.uniform(0.8, 1.2)

                if curr_ap > prev_ap:
                    buy_a  = volume_a * random.uniform(0.55, 0.75)
                    sell_a = volume_a - buy_a
                elif curr_ap < prev_ap:
                    sell_a = volume_a * random.uniform(0.55, 0.75)
                    buy_a  = volume_a - sell_a
                else:
                    buy_a = sell_a = volume_a * 0.5

                self.asset_volume_history[a].append(volume_a)
                self.asset_buy_volume_history[a].append(buy_a)
                self.asset_sell_volume_history[a].append(sell_a)

                hist_a = self.asset_volume_history[a]
                self.asset_rolling_vol[a] = sum(hist_a[-20:]) / 20 if len(hist_a) >= 20 else 0.0

                imb_a   = buy_a - sell_a
                denom_a = buy_a + sell_a
                self.asset_vol_imbalance[a] = imb_a
                self.asset_pressure[a]      = imb_a / denom_a if denom_a > 0 else 0.0

            # --- Attach per-asset features to the state object -----------
            for a in self.assets:
                if not self.asset_prices[a]:
                    continue
                setattr(state, f"{a}_price",        self.asset_prices[a][-1])
                setattr(state, f"{a}_vol",           self.asset_vol[a])
                setattr(state, f"{a}_mom_5",         self.asset_mom_5[a])
                setattr(state, f"{a}_mom_20",        self.asset_mom_20[a])
                setattr(state, f"{a}_mom_50",        self.asset_mom_50[a])
                setattr(state, f"{a}_rolling_vol",   self.asset_rolling_vol[a])
                setattr(state, f"{a}_vol_imbalance", self.asset_vol_imbalance[a])
                setattr(state, f"{a}_pressure",      self.asset_pressure[a])

            # --- Crypto-regime features -----------------------------------
            # All five assets must have at least one price before we compute;
            # guard with a short-circuit so step-1 stays safe.
            if all(self.asset_prices[a] for a in self.assets):
                _btc = self.asset_prices["BTC"][-1]
                _eth  = self.asset_prices["ETH"][-1]
                _sol  = self.asset_prices["SOL"][-1]
                _xrp  = self.asset_prices["XRP"][-1]
                _lnk  = self.asset_prices["LINK"][-1]
                _xlm  = self.asset_prices["XLM"][-1]
                _hbar = self.asset_prices["HBAR"][-1]
                _avax = self.asset_prices["AVAX"][-1]

                # --- BTC dominance (synthetic market caps) ----------------
                _btc_mc  = _btc  * 19_000_000
                _eth_mc  = _eth  * 120_000_000
                _sol_mc  = _sol  * 440_000_000
                _xrp_mc  = _xrp  * 50_000_000_000
                _lnk_mc  = _lnk  * 587_000_000
                _xlm_mc  = _xlm  * 25_000_000_000
                _hbar_mc = _hbar * 35_000_000_000
                _avax_mc = _avax * 350_000_000
                _total_mc = (_btc_mc + _eth_mc + _sol_mc + _xrp_mc
                             + _lnk_mc + _xlm_mc + _hbar_mc + _avax_mc)
                self.btc_dominance = _btc_mc / _total_mc if _total_mc > 0 else 0.0

                # --- Cross-asset ratios -----------------------------------
                self.eth_btc_ratio    = _eth / _btc if _btc > 0 else 0.0
                self.sol_btc_strength = _sol / _btc if _btc > 0 else 0.0
                self.sol_eth_strength = _sol / _eth if _eth > 0 else 0.0

                # --- Altseason index (three-component composite) ----------
                # Component 1: ETH/BTC ratio (already computed above)
                _eth_vs_btc = self.eth_btc_ratio

                # Component 2: altcoin volume share (all non-BTC, non-ETH alts)
                # Use most-recent step volume from volume history; default 0.
                _bvol  = self.asset_volume_history["BTC"][-1]  if self.asset_volume_history["BTC"]  else 0.0
                _evol  = self.asset_volume_history["ETH"][-1]  if self.asset_volume_history["ETH"]  else 0.0
                _svol  = self.asset_volume_history["SOL"][-1]  if self.asset_volume_history["SOL"]  else 0.0
                _xvol  = self.asset_volume_history["XRP"][-1]  if self.asset_volume_history["XRP"]  else 0.0
                _lvol  = self.asset_volume_history["LINK"][-1] if self.asset_volume_history["LINK"] else 0.0
                _xlvol = self.asset_volume_history["XLM"][-1]  if self.asset_volume_history["XLM"]  else 0.0
                _hvol  = self.asset_volume_history["HBAR"][-1] if self.asset_volume_history["HBAR"] else 0.0
                _avvol = self.asset_volume_history["AVAX"][-1] if self.asset_volume_history["AVAX"] else 0.0
                _total_vol = _bvol + _evol + _svol + _xvol + _lvol + _xlvol + _hvol + _avvol
                _alt_vol   = _svol + _xvol + _lvol + _xlvol + _hvol + _avvol
                _alt_vol_share = _alt_vol / _total_vol if _total_vol > 0 else 0.0

                # Component 3: fraction of alts beating BTC this step
                _btc_ret = self.asset_returns["BTC"][-1] if self.asset_returns["BTC"] else 0.0
                _alts_outperform = sum(
                    1 for a in ["SOL", "XRP", "LINK", "ETH", "XLM", "HBAR", "AVAX"]
                    if self.asset_returns[a] and self.asset_returns[a][-1] > _btc_ret
                )
                _alts_outperform_norm = _alts_outperform / 7

                self.altseason_index = (
                    0.4 * _eth_vs_btc
                    + 0.3 * _alt_vol_share
                    + 0.3 * _alts_outperform_norm
                )

                # --- Volatility regime: avg BTC + ETH realized vol -------
                self.vol_regime = (self.asset_vol["BTC"] + self.asset_vol["ETH"]) / 2.0

                # --- Liquidity regime: mean rolling vol across all assets -
                _rvols = [self.asset_rolling_vol[a] for a in self.assets if self.asset_rolling_vol[a] > 0]
                self.liq_regime = sum(_rvols) / len(_rvols) if _rvols else 0.0

            # --- Internal regime scoring (used by AUTO mode) --------------
            # Scores are computed every step so they're always up-to-date
            # and can be exposed to the state for diagnostics.
            _alt     = self.altseason_index
            _vol     = self.vol_regime
            _liq     = self.liq_regime
            _btc_dom = self.btc_dominance

            _bull_score = 0.0
            _bear_score = 0.0

            # Bullish signals
            if _alt > 0.5:
                _bull_score += 1.0   # strong altcoin momentum
            if _vol < 0.01:
                _bull_score += 1.0   # low cross-market volatility
            if _btc_dom < 0.45:
                _bull_score += 1.0   # capital rotating into alts
            if _liq > 0.5 * max(1e-6, _liq):   # placeholder: any liquidity
                _bull_score += 0.5

            # Bearish signals
            if _vol > 0.02:
                _bear_score += 1.0   # elevated cross-market stress
            if _btc_dom > 0.55:
                _bear_score += 1.0   # capital fleeing to BTC (risk-off)
            if _alt < 0.2:
                _bear_score += 1.0   # weak altcoin momentum
            if _liq < 0.3 * max(1e-6, _liq):   # placeholder: thin liquidity
                _bear_score += 0.5

            if _bull_score - _bear_score > 0.5:
                self.internal_regime = +1
            elif _bear_score - _bull_score > 0.5:
                self.internal_regime = -1
            else:
                self.internal_regime = 0

            # Attach regime features to state for featurize_state() to consume
            state.btc_dominance    = self.btc_dominance
            state.eth_btc_ratio    = self.eth_btc_ratio
            state.sol_btc_strength = self.sol_btc_strength
            state.sol_eth_strength = self.sol_eth_strength
            state.altseason_index  = self.altseason_index
            state.vol_regime       = self.vol_regime
            state.liq_regime       = self.liq_regime
            state.macro_regime     = self.macro_regime
            state.internal_regime  = self.internal_regime

            # --- Multi-asset rotation engine ------------------------------
            # Derives relative strength for all alts vs BTC/ETH anchors
            # using existing price buffers — no new data sources.
            _rot_assets = ["SOL", "XRP", "LINK", "ETH", "XLM", "HBAR", "AVAX"]
            _btc_prices = self.asset_prices["BTC"]
            _eth_prices = self.asset_prices["ETH"]

            self.rotation_score = {}

            for _ra in _rot_assets:
                _rap = self.asset_prices[_ra]
                _rap_n = len(_rap)

                # Short-term momentum (5-step percentage return)
                _rm5_a   = (_rap[-1] - _rap[-5])   / max(_rap[-5],   1e-6) if _rap_n >= 5  else 0.0
                _rm5_btc = (_btc_prices[-1] - _btc_prices[-5]) / max(_btc_prices[-5], 1e-6) \
                           if len(_btc_prices) >= 5 else 0.0

                # Medium-term momentum (20-step percentage return)
                _rm20_a   = (_rap[-1] - _rap[-20])  / max(_rap[-20],  1e-6) if _rap_n >= 20 else 0.0
                _rm20_btc = (_btc_prices[-1] - _btc_prices[-20]) / max(_btc_prices[-20], 1e-6) \
                            if len(_btc_prices) >= 20 else 0.0

                # Strength ratios vs anchor assets
                _vs_btc = _rap[-1] / max(_btc_prices[-1], 1e-6) if _btc_prices else 0.0
                _vs_eth = _rap[-1] / max(_eth_prices[-1], 1e-6) if _eth_prices else 0.0

                _rscore = 0.0
                if _rm5_a > 0:               _rscore += 1.0   # positive 5-step mom
                if _rm5_a > _rm5_btc:        _rscore += 0.5   # outperforming BTC (5-step)
                if _rm20_a > 0:              _rscore += 1.0   # positive 20-step mom
                if _rm20_a > _rm20_btc:      _rscore += 0.5   # outperforming BTC (20-step)
                if _vs_btc > 1.0:            _rscore += 0.5   # priced above BTC ratio
                if _vs_eth > 1.0:            _rscore += 0.5   # priced above ETH ratio
                _rscore += 0.5 * self.altseason_index          # altseason tailwind

                self.rotation_score[_ra] = _rscore

            setattr(state, "rotation_score", self.rotation_score)

            # Rotation feedback: only on meaningful score shifts (>1.0)
            for _ra in _rot_assets:
                _rold = self.prev_rotation_score.get(_ra)
                _rnew = self.rotation_score[_ra]
                if _rold is not None and abs(_rnew - _rold) > 1.0:
                    _rmsg = (
                        f"Rotation shift: {_ra} score changed "
                        f"{_rold:.2f} → {_rnew:.2f}"
                    )
                    self.macro_messages.append(_rmsg)
                    print(_rmsg)

            self.prev_rotation_score = self.rotation_score.copy()

            # --- Blow-off top detection -----------------------------------
            # Scores late-cycle, unsustainable acceleration using existing
            # regime features — no new data sources required.
            _bot_alt    = self.altseason_index
            _bot_vol    = self.vol_regime
            _bot_liq    = self.liq_regime
            _bot_btcdom = self.btc_dominance
            _bot_score  = 0.0

            if _bot_alt > 0.70:   _bot_score += 1.0   # extreme altseason
            if _bot_alt > 0.85:   _bot_score += 1.0   # very extreme
            if _bot_vol > 0.015:  _bot_score += 1.0   # rising vol in strength
            if _bot_vol > 0.025:  _bot_score += 1.0   # severe instability
            if _bot_liq < 0.30:   _bot_score += 1.0   # liquidity exhaustion
            if _bot_btcdom < 0.40: _bot_score += 1.0  # alt blow-off rotation
            if _bot_btcdom < 0.35: _bot_score += 1.0  # extreme alt dominance

            if _bot_score >= 4:
                self.top_risk = 2   # blow-off
            elif _bot_score >= 2:
                self.top_risk = 1   # early warning
            else:
                self.top_risk = 0   # normal

            setattr(state, "top_risk", self.top_risk)

            # Blow-off feedback: print only on transitions
            if self.top_risk != self.prev_top_risk:
                if self.top_risk == 1:
                    _tmsg = "Top-risk rising: early blow-off conditions detected."
                elif self.top_risk == 2:
                    _tmsg = "Blow-off top conditions detected — de-risking recommended."
                else:
                    _tmsg = "Top-risk normalized."
                self.macro_messages.append(_tmsg)
                print(_tmsg)
            self.prev_top_risk = self.top_risk

            # --- Dynamic exposure caps ------------------------------------
            # Per-asset caps that adapt each step to volatility, liquidity,
            # altseason momentum, and the effective macro regime.
            _base_caps = {
                "SOL":  1.0,
                "XRP":  0.8,
                "LINK": 0.7,
                "ETH":  0.6,
                "BTC":  0.3,
                "XLM":  0.6,
                "HBAR": 0.6,
                "AVAX": 0.7,
            }
            _alt = self.altseason_index
            _vol = self.vol_regime
            _liq = self.liq_regime
            _regime = self.internal_regime if self.macro_regime == 2 else self.macro_regime

            self.dynamic_caps = {}
            for _asset, _base in _base_caps.items():
                _cap = _base

                if _regime > 0:
                    _cap *= (1.0 + 0.5 * _alt)
                    if _vol > 0.02:
                        _cap *= 0.7
                    if _liq > 0.5:
                        _cap *= 1.2
                elif _regime < 0:
                    _cap *= 0.5
                    if _vol > 0.02:
                        _cap *= 0.5
                    if _liq < 0.3:
                        _cap *= 0.7
                else:
                    if _vol > 0.02:
                        _cap *= 0.7
                    if _liq > 0.5:
                        _cap *= 1.1

                # Tighten caps during blow-off risk (applied before clamp)
                if self.top_risk == 1:
                    _cap *= 0.8   # early warning: moderate reduction
                elif self.top_risk == 2:
                    _cap *= 0.5   # blow-off: de-risk hard

                # Rotation tilt: leaders get up to +20% cap
                # BTC is the anchor so has no rotation score entry (returns 0.0)
                _rot_raw  = self.rotation_score.get(_asset, 0.0)
                _rot_norm = min(1.0, _rot_raw / 4.0)   # normalise to [0, 1]
                _cap *= (1.0 + 0.2 * _rot_norm)

                # Clamp to [0.1, 1.5 × base]
                self.dynamic_caps[_asset] = max(0.1, min(1.5 * _base, _cap))

            setattr(state, "dynamic_caps", self.dynamic_caps)

            # --- Dynamic cap change feedback ------------------------------
            for _asset in self.assets:
                _old = self.prev_dynamic_caps.get(_asset)
                _new = self.dynamic_caps[_asset]
                if _old is not None and abs(_new - _old) > 0.2 * _old:
                    _cmsg = (
                        f"Dynamic cap shift for {_asset}: "
                        f"{_old:.2f} → {_new:.2f} "
                        f"(regime={_regime:+d}, vol={_vol:.3f}, alt={_alt:.3f})"
                    )
                    self.macro_messages.append(_cmsg)
                    print(_cmsg)

            self.prev_dynamic_caps = self.dynamic_caps.copy()

            # --- Macro feedback: signal / manual-call conflict detection --
            _msg = None
            if self.macro_regime in (-1, 0, +1):
                # Manual mode: warn when internal regime disagrees
                if self.internal_regime != self.macro_regime:
                    _msg = (
                        f"Signals conflict with your macro call: "
                        f"you set {self.macro_regime:+d}, "
                        f"signals suggest {self.internal_regime:+d}."
                    )
            elif self.macro_regime == 2:
                # AUTO mode: warn when signals are uncertain / evenly split
                if abs(_bull_score - _bear_score) < 0.5:
                    _msg = (
                        f"AUTO mode: signals uncertain or mixed "
                        f"(bull={_bull_score:.1f}, bear={_bear_score:.1f})."
                    )
            if _msg:
                # Store and print only on transitions (new or changed message)
                # so the log reflects regime changes, not per-step repetitions.
                if _msg != self._last_macro_message:
                    self.macro_messages.append(_msg)
                    print(f"[MacroFeedback] {_msg}")
                    self._last_macro_message = _msg
            else:
                # Reset: the same message will re-fire if conditions return
                self._last_macro_message = None

            # --- Deliver queued orders that are due ----------------------
            # Use the *current* state for the risk check so a delayed order
            # is evaluated against up-to-date market conditions.
            due = [
                (s, a, i) for (s, a, i) in self.latency_queue
                if s <= current_step
            ]
            self.latency_queue = [
                (s, a, i) for (s, a, i) in self.latency_queue
                if s > current_step
            ]
            for _, agent, intent in due:
                if self.risk_manager.approve_order(agent, intent, state):
                    self.broker.submit_order(agent, intent, state)

            # --- Agent decisions → latency queue -------------------------
            step_actions = {}
            for agent in self.agents:
                action = agent.decide(state)
                step_actions[agent] = action

                if isinstance(agent, ReinforcementLearningTrader):
                    # RL agent manages exposure directly via execute_exposure();
                    # it does not go through the latency queue or order book.
                    pass
                elif action != "hold":
                    intent = OrderIntent(side=action, quantity=1, price=price)
                    deliver_step = current_step + agent.latency
                    self.latency_queue.append((deliver_step, agent, intent))

            # --- End-of-step settlement via broker -----------------------
            self.broker.fill_resting_orders(price)

            # --- RL agent: ramp portfolio exposures toward targets --------
            # Store the current state so apply_microstructure() can read
            # state.equity_vol / pressure / rolling_vol via self.current_state.
            self.current_state = state

            # execute_portfolio_exposure() returns total txn cost across all
            # assets so the simulation can subtract it from the reward.
            rl_txn_costs    = {}
            rl_turnovers    = {}
            for agent in self.agents:
                if isinstance(agent, ReinforcementLearningTrader):
                    # Turnover = Σ |target − current| before the broker ramps.
                    # Computed here so exposure-change caps don't hide it.
                    rl_turnovers[agent] = sum(
                        abs(agent.target_exposures[a] - agent.positions[a])
                        for a in agent.assets
                    )
                    # Build per-asset price dict from the simulation's buffers.
                    _asset_prices = {
                        a: (self.asset_prices[a][-1] if self.asset_prices[a] else 100.0)
                        for a in agent.assets
                    }
                    cost = self.broker.execute_portfolio_exposure(
                        agent, _asset_prices, self.apply_microstructure,
                        simulation=self,
                    )
                    rl_txn_costs[agent] = cost

            # --- Tracking + actor-critic update --------------------------
            for agent in self.agents:
                agent.update_last_price(state.mid_price)

                if isinstance(agent, ReinforcementLearningTrader):
                    # Portfolio unrealized PnL: sum exposure × price per asset.
                    _apx = {
                        a: (self.asset_prices[a][-1] if self.asset_prices[a] else 100.0)
                        for a in agent.assets
                    }
                    agent.unrealized_pnl = sum(
                        agent.positions[a] * _apx[a] for a in agent.assets
                    )
                else:
                    agent.update_unrealized_pnl(price)

                agent.record_equity()

                if isinstance(agent, ReinforcementLearningTrader):
                    new_encoded = agent.featurize_state(state, agent)
                    new_equity = agent.balance + agent.unrealized_pnl

                    # --- Equity peak / drawdown ---------------------------
                    self.equity_history.append(new_equity)
                    if self.equity_peak is None:
                        self.equity_peak = new_equity
                    if new_equity > self.equity_peak:
                        self.equity_peak = new_equity
                    state.drawdown = max(0.0, self.equity_peak - new_equity)

                    # --- Rolling equity-return volatility -----------------
                    # Computed over the last vol_window 1-step equity returns.
                    # Returns < 2 data points → vol is zero (no variance yet).
                    _vol_window = 20
                    if len(self.equity_history) >= 2:
                        _returns = []
                        for _i in range(1, len(self.equity_history)):
                            _prev = self.equity_history[_i - 1]
                            _curr = self.equity_history[_i]
                            if _prev != 0:
                                _returns.append((_curr - _prev) / _prev)
                        if len(_returns) >= _vol_window:
                            _slice = _returns[-_vol_window:]
                            _mean = sum(_slice) / len(_slice)
                            _var  = sum((r - _mean) ** 2 for r in _slice) / max(len(_slice) - 1, 1)
                            state.equity_vol = math.sqrt(_var)
                        else:
                            state.equity_vol = 0.0
                    else:
                        state.equity_vol = 0.0

                    if agent.prev_state is not None:
                        # --- Regime-aware portfolio reward ----------------
                        pnl_reward    = new_equity - agent.prev_equity
                        step_txn_cost = rl_txn_costs.get(agent, 0.0)
                        turnover      = rl_turnovers.get(agent, 0.0)

                        # Risk adjustments that apply regardless of regime
                        risk_adj   = 0.1 * state.drawdown + 0.05 * state.equity_vol
                        step_return = pnl_reward - risk_adj
                        base_reward = step_return - step_txn_cost

                        macro = self.macro_regime
                        # AUTO mode: ignore manual call, use internally
                        # derived regime so reward shaping is fully signal-driven.
                        macro_effective = (
                            self.internal_regime if macro == 2 else macro
                        )
                        alt = getattr(state, "altseason_index", 0.0)
                        vol = getattr(state, "vol_regime", 0.0)

                        if macro_effective > 0:
                            # Bull: encourage upside capture, light turnover penalty
                            turnover_penalty = 0.001 * turnover
                            reward = base_reward * (1.0 + 0.5 * alt) - turnover_penalty
                        elif macro_effective < 0:
                            # Bear: heavily penalize drawdowns and churn
                            turnover_penalty = 0.005 * turnover
                            downside = min(pnl_reward, 0.0)
                            reward = (
                                base_reward
                                - 2.0 * abs(downside) * (1.0 + 5.0 * vol)
                                - turnover_penalty
                            )
                        else:
                            # Neutral / AUTO-neutral: balanced middle ground
                            turnover_penalty = 0.003 * turnover
                            reward = base_reward - turnover_penalty

                        # --- Blow-off top reward adjustment ---------------
                        # Discourage adding exposure into blow-off conditions.
                        if self.top_risk == 1:
                            reward -= 0.5 * turnover
                        elif self.top_risk == 2:
                            reward -= 1.0 * turnover
                            if step_return < 0:
                                reward -= 2.0 * abs(step_return)

                        # --- Rotation leader bonus ------------------------
                        # Small positive signal toward overweighting leaders;
                        # averaged across all scored alts so it's portfolio-level.
                        if self.rotation_score:
                            _rot_bonus = 0.01 * (
                                sum(self.rotation_score.values())
                                / len(self.rotation_score)
                            )
                            reward += _rot_bonus

                        # --- Critic: N-step replay buffer update ----------
                        done = (current_step == SIMULATION_STEPS - 1)
                        agent.add_experience(
                            agent.prev_state,
                            agent.prev_action,
                            reward,
                            new_encoded,
                            done,
                        )
                        agent.replay()

                        # --- Actor: online one-step advantage update ------
                        # Resamples a fresh action on prev_state to get a
                        # valid computation graph anchored in the current
                        # (just-updated) actor parameters.
                        agent.update_actor(
                            agent.prev_state,
                            reward,
                            new_encoded,
                        )

                    agent.prev_state  = new_encoded
                    agent.prev_action = step_actions[agent]
                    agent.prev_equity = new_equity

            # --- Global risk check ---------------------------------------
            self.risk_manager.check_global_risk(self.agents)

            # --- Histories -----------------------------------------------
            self.price_history.append(price)
            self.regime_history.append(tick.regime)

        return self.broker.trade_log, self.agents, self.price_history, self.regime_history
