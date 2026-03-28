import os
import sys
import time
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.simulation import Simulation
from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.rl_agent import ReinforcementLearningTrader
from agents.market_agent import MarketAgent
from market_data.data_source import SimulatedDataSource
from broker.broker import SimulatedBroker, LiveBroker, PaperBroker, run_live_trading_loop
from exchange.exchange import Exchange
from risk.risk_manager import RiskManager
from config.config import INITIAL_BALANCE, MARKET_START_PRICE
from archive.trade_archive import TradeArchive
from strategies.bull_bear_trader import BullBearRotationalTrader

# ---------------------------------------------------------------------------
# Mode switch — "sim" runs the local simulation; "live" starts the Kraken
# live trading loop (requires KRAKEN_API_KEY / KRAKEN_API_SECRET secrets).
# Override via BOT_MODE environment variable: BOT_MODE=sim python3 main.py
# ---------------------------------------------------------------------------
MODE = os.environ.get("BOT_MODE", "live")

NUM_EPISODES = 10

REGIME_COLORS = {
    "bull":     "green",
    "bear":     "red",
    "high_vol": "orange",
    "low_vol":  "blue",
    "crash":    "purple",
}


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def shade_regimes(ax, regime_history):
    seen = set()
    i = 0
    while i < len(regime_history):
        current = regime_history[i]
        j = i
        while j < len(regime_history) and regime_history[j] == current:
            j += 1
        label = current if current not in seen else None
        seen.add(current)
        ax.axvspan(i, j, color=REGIME_COLORS[current], alpha=0.15, label=label)
        i = j


def plot_price_chart(price_history, regime_history):
    fig, ax = plt.subplots(figsize=(12, 5))
    shade_regimes(ax, regime_history)
    ax.plot(price_history, color="black", linewidth=1.2, label="Price")
    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r.replace("_", " ").title())
        for r in REGIME_COLORS if r in regime_history
    ]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="black", label="Price")],
              loc="upper left", fontsize=8)
    ax.set_title("Price History with Regime Shading (final episode)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("price_chart.png", dpi=150)
    plt.close()
    print("Saved: price_chart.png")


def plot_equity_curves(agents):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    for agent, color in zip(agents, colors):
        ax.plot(agent.equity_history, label=agent.name, color=color, linewidth=1.5)
    if agents:
        ax.axhline(y=agents[0].equity_history[0], color="gray", linestyle="--",
                   linewidth=0.8, label="Starting equity")
    ax.set_title("Agent Equity Curves (final episode)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Equity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curves.png", dpi=150)
    plt.close()
    print("Saved: equity_curves.png")


def plot_drawdowns(agents):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    for agent, color in zip(agents, colors):
        equity = agent.equity_history
        peak = equity[0]
        dd_series = []
        for v in equity:
            peak = max(peak, v)
            dd_series.append((peak - v) / peak if peak > 0 else 0)
        ax.plot(dd_series, label=agent.name, color=color, linewidth=1.5)
    ax.set_title("Agent Drawdowns (final episode)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("drawdowns.png", dpi=150)
    plt.close()
    print("Saved: drawdowns.png")


def plot_rl_qtable_growth(qtable_sizes):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(qtable_sizes) + 1), qtable_sizes,
            marker="o", color="crimson", linewidth=1.5)
    ax.set_title("RLTrader Q-table Growth Across Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Unique States in Q-table")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rl_qtable_growth.png", dpi=150)
    plt.close()
    print("Saved: rl_qtable_growth.png")


def print_risk_summary(risk_manager):
    summary = risk_manager.episode_summary()
    rejected = summary["rejected_per_agent"]
    locked = summary["drawdown_locked"]
    gks = summary["global_kill_switch"]
    etf_skips = summary.get("etf_skips_per_agent", {})
    strategy_holds = summary.get("strategy_holds_per_agent", {})

    # ---- Risk violations (blocks, locks, kill-switch) ----
    total_blocked = sum(rejected.values())
    if total_blocked == 0 and not locked and not gks:
        print("  [Risk] No limits hit this episode.")
    else:
        parts = [f"{name}={n}" for name, n in rejected.items() if n > 0]
        print(f"  [Risk] Orders blocked: {', '.join(parts) if parts else 'none'}")
        if locked:
            print(f"  [Risk] Drawdown-locked: {', '.join(locked)}")
        if gks:
            print("  [Risk] *** GLOBAL KILL-SWITCH triggered ***")

    # ---- Valid strategy events (not risk violations) ----
    if etf_skips:
        parts = [f"{name}={n}" for name, n in etf_skips.items()]
        print(f"  [Strategy] ETF skips (neutral/threshold/timeout): {', '.join(parts)}")
    if strategy_holds:
        parts = [f"{name}={n}" for name, n in strategy_holds.items()]
        print(f"  [Strategy] Position holds (fee-hurdle/neutral):   {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _broker_mode_label(broker) -> str:
    if isinstance(broker, PaperBroker):
        return "PAPER (synthetic fills, CSV log)"
    if isinstance(broker, LiveBroker) and broker._sandbox_mode:
        return "SANDBOX (validate=true, no fills)"
    return "LIVE (real orders)"


def run_live():
    """
    Launch the Kraken live trading loop.

    Three-tier broker selection (set in Replit Secrets):
    ┌──────────────────────────────────────────────────────────────────┐
    │  USE_PAPER_BROKER=true  (default)                                │
    │  → PaperBroker: synthetic fills, CSV log, zero API write calls   │
    │                                                                  │
    │  USE_PAPER_BROKER=false + KRAKEN_SANDBOX=true                    │
    │  → LiveBroker with validate=true: Kraken validates, no fills     │
    │                                                                  │
    │  USE_PAPER_BROKER=false + KRAKEN_SANDBOX=false                   │
    │  → LiveBroker full live: real orders, real fills                 │
    └──────────────────────────────────────────────────────────────────┘

    Strategy selection (set in Replit Secrets):
    ┌──────────────────────────────────────────────────────────────────┐
    │  USE_BULL_BEAR_TRADER=true                                       │
    │  → BullBearRotationalTrader: phase-driven bull/bear cycle bot   │
    │                                                                  │
    │  USE_BULL_BEAR_TRADER=false (default)                            │
    │  → ReinforcementLearningTrader / MA crossover (original)         │
    └──────────────────────────────────────────────────────────────────┘
    """
    import os as _os
    import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    _ET = _ZoneInfo("America/New_York")

    use_paper = _os.environ.get("USE_PAPER_BROKER", "true").strip().lower() \
                not in ("false", "0", "no")

    use_bull_bear = _os.environ.get("USE_BULL_BEAR_TRADER", "false").strip().lower() \
                    in ("true", "1", "yes")

    print("[MAIN] Starting live trading loop...")

    if use_paper:
        archive = TradeArchive()
        broker = PaperBroker(initial_cash=INITIAL_BALANCE, trade_archive=archive)
    else:
        archive = TradeArchive()   # persist phase/rotation events even in live mode
        broker = LiveBroker(dry_run=False)

    print(f"[MAIN] Broker mode: {_broker_mode_label(broker)}")

    # Futures wallet — paper-simulated when ENABLE_FUTURES=False (default)
    if broker.futures_paper_mode:
        print("[MAIN] Futures overlay: PAPER mode — simulating collateral from spot equity.")
    else:
        print("[MAIN] Futures overlay: LIVE mode — fetching real collateral from futures.kraken.com.")
    broker.fetch_futures_wallet()

    # ---------------------------------------------------------------
    # Strategy selection
    # ---------------------------------------------------------------
    if use_bull_bear:
        print("[MAIN] Strategy: BullBearRotationalTrader (phase-driven bull/bear cycle)")
        bull_bear_trader = BullBearRotationalTrader(
            broker=broker,
            archive=archive,
            initial_phase="accumulation",
        )

        # Tag paper fills with the strategy name so the archive and CSV log
        # attribute trades to "BullBearTrader" instead of an empty string.
        if isinstance(broker, PaperBroker):
            broker.set_strategy_name("BullBearTrader")

        # Sync fee tier if using live broker
        if isinstance(broker, LiveBroker) and not isinstance(broker, PaperBroker):
            broker.sync_fee_tier_from_kraken()

        # Initial account sync
        state = broker.sync_live_account_state()
        if state is None:
            print("[MAIN] Failed initial account sync — aborting")
            if isinstance(broker, PaperBroker):
                broker.close()
            return

        balances, positions = state
        print(f"[MAIN] Initial balances: {balances}")
        print(
            "[MAIN] Commands: 'S' + Enter = paper summary  |  'P' = phase status  |  Ctrl-C = stop"
        )

        SUMMARY_INTERVAL_SEC      = 900    # 15 minutes
        BALANCE_SYNC_INTERVAL_SEC = 7200   # 2 hours
        _last_summary_ts          = time.time()
        _last_balance_sync_ts     = time.time()
        _tick                     = 0

        # Track USD balance across re-syncs so fresh cash deposits are detected
        _known_usd = float(balances.get("ZUSD", 0.0))

        # ETF-first priority allocation on startup (if cash is available and
        # the regime can be inferred from the bull_bear_trader's current phase)
        if hasattr(broker, "run_etf_priority_allocation") and _known_usd > 0:
            _startup_regime = {"cycle_phase": 1}  # neutral default (expansion/bull)
            try:
                _phase_map = {
                    "accumulation":   {"cycle_phase": 0},
                    "bull_alt_season":{"cycle_phase": 1},
                    "alt_cascade":    {"cycle_phase": 2},
                    "bear_market":    {"cycle_phase": 3},
                }
                _startup_regime = _phase_map.get(
                    getattr(bull_bear_trader, "phase", "bull_alt_season"),
                    {"cycle_phase": 1},
                )
            except Exception:
                pass
            _ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[MAIN] [{_ts}] Startup ETF priority allocation: cash=${_known_usd:.2f}  regime={_startup_regime}")
            broker.run_etf_priority_allocation(
                available_cash = _known_usd,
                regime         = _startup_regime,
            )

        import queue, threading

        _cmd_queue: queue.Queue = queue.Queue()

        def _stdin_reader():
            try:
                for raw_line in iter(sys.stdin.readline, ""):
                    cmd = raw_line.strip().upper()
                    if cmd:
                        _cmd_queue.put(cmd)
            except Exception:
                pass

        _stdin_thread = threading.Thread(target=_stdin_reader, daemon=True, name="stdin-reader")
        _stdin_thread.start()

        try:
            while True:
                if broker.kill_switch:
                    print("[MAIN] Kill switch active — exiting loop")
                    break

                # Optional auto-shutdown at 4:30 PM ET.
                # Disabled by default so the bot runs 24/7 (crypto never closes).
                # Enable with: AUTO_SHUTDOWN_ET=true ./run_sandbox.sh
                _auto_shutdown = _os.environ.get(
                    "AUTO_SHUTDOWN_ET", "false").strip().lower() in ("true", "1", "yes")
                if _auto_shutdown:
                    _now_et = _datetime.datetime.now(_ET)
                    if (_now_et.weekday() < 5 and
                            _now_et >= _now_et.replace(
                                hour=16, minute=30, second=0, microsecond=0)):
                        print(
                            "[MAIN] 16:30 ET reached — end-of-day auto-shutdown.\n"
                            "[MAIN] All trades recorded. EOD report will be saved now."
                        )
                        break

                # Keyboard commands
                try:
                    cmd = _cmd_queue.get_nowait()
                    if cmd == "S":
                        if isinstance(broker, PaperBroker):
                            broker.print_summary()
                        else:
                            print("[MAIN] Summary only available in PAPER mode.")
                    elif cmd == "P":
                        bull_bear_trader.print_status()
                    else:
                        print(f"[MAIN] Unknown command: {cmd!r}  (known: S, P)")
                except queue.Empty:
                    pass

                # Auto summary every 15 minutes
                _now = time.time()
                if _now - _last_summary_ts >= SUMMARY_INTERVAL_SEC:
                    print("[MAIN] 15-minute auto-summary:")
                    if isinstance(broker, PaperBroker):
                        broker.print_summary()
                    bull_bear_trader.print_status()
                    _last_summary_ts = _now

                # Auto balance re-sync every 2 hours (live mode only — paper
                # tracks its own synthetic cash and does not need this)
                if not isinstance(broker, PaperBroker) and _now - _last_balance_sync_ts >= BALANCE_SYNC_INTERVAL_SEC:
                    _ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    try:
                        _refreshed = broker.fetch_live_balances()
                        if _refreshed is not None:
                            _usd = float(_refreshed.get("ZUSD", 0.0))
                            print(f"[MAIN] [{_ts}] Balance re-sync: USD available = ${_usd:,.2f}")

                            # Detect fresh cash (deposit since last sync)
                            _FRESH_CASH_THRESHOLD = 1.0   # $1 minimum to avoid floating-point noise
                            if _usd > _known_usd + _FRESH_CASH_THRESHOLD:
                                _fresh = _usd - _known_usd
                                print(
                                    f"[MAIN] [{_ts}] Fresh cash detected: +${_fresh:.2f} "
                                    f"(${_known_usd:.2f} → ${_usd:.2f})"
                                )
                                # ETF-first priority allocation on the new funds
                                if hasattr(broker, "run_etf_priority_allocation"):
                                    _phase_map = {
                                        "accumulation":   {"cycle_phase": 0},
                                        "bull_alt_season":{"cycle_phase": 1},
                                        "alt_cascade":    {"cycle_phase": 2},
                                        "bear_market":    {"cycle_phase": 3},
                                    }
                                    _current_regime = _phase_map.get(
                                        getattr(bull_bear_trader, "phase", "bull_alt_season"),
                                        {"cycle_phase": 1},
                                    )
                                    print(
                                        f"[MAIN] [{_ts}] Triggering ETF priority allocation "
                                        f"on fresh cash: ${_usd:.2f}  regime={_current_regime}"
                                    )
                                    broker.run_etf_priority_allocation(
                                        available_cash = _usd,
                                        regime         = _current_regime,
                                    )

                            _known_usd = _usd
                        else:
                            print(f"[MAIN] [{_ts}] Balance re-sync failed — Kraken API unavailable, keeping previous balance")
                    except Exception as _e:
                        print(f"[MAIN] [{_ts}] Balance re-sync error: {_e} — continuing")
                    _last_balance_sync_ts = _now

                # Fetch live prices from Kraken public API
                broker.fetch_live_prices()
                prices = dict(broker.live_prices)

                if prices:
                    bull_bear_trader.step(prices)

                    # ---- Live performance display --------------------------------
                    status = bull_bear_trader.status_summary()
                    _tick += 1

                    # Build equity/P&L string for paper mode
                    if isinstance(broker, PaperBroker):
                        equity  = broker.compute_total_equity()
                        rpnl    = broker.paper_realized_pnl
                        upnl    = broker.get_unrealized_pnl()
                        n_fills = len(broker.paper_trade_history)
                        pnl_str = (
                            f"  equity=${equity:.2f}"
                            f"  rpnl=${rpnl:+.2f}"
                            f"  upnl=${upnl:+.2f}"
                            f"  fills={n_fills}"
                        )
                    else:
                        pnl_str = ""

                    btc_price = prices.get("BTC", 0.0)
                    print(
                        f"[MAIN #{_tick:04d}] "
                        f"BTC=${btc_price:,.0f}  "
                        f"phase={status['phase']}  "
                        f"conf={status['btc_confidence']:.2f}  "
                        f"top={status['market_topping']}  "
                        f"rec={status['recovering']}"
                        f"{pnl_str}"
                    )

                    # Show open positions (condensed, only when non-empty)
                    if status["positions"]:
                        pos_str = "  ".join(
                            f"{a}={v*100:.1f}%"
                            for a, v in status["positions"].items()
                        )
                        print(f"         positions: {pos_str}")

                broker.heartbeat()
                broker.record_health_metrics()
                broker.daily_rollover()
                broker.alerting_loop()

                time.sleep(1.0)

        except KeyboardInterrupt:
            print("[MAIN] KeyboardInterrupt — shutting down cleanly")
            if isinstance(broker, PaperBroker):
                broker.print_summary()
            bull_bear_trader.print_status()
        except Exception as e:
            print(f"[MAIN] Unhandled exception: {e}")
            broker.trigger_kill_switch(f"Main loop exception: {e}")
        finally:
            if isinstance(broker, PaperBroker):
                broker.close()

        return  # end of bull_bear path

    # ---------------------------------------------------------------
    # Original RL / MA strategy path
    # ---------------------------------------------------------------
    agent = ReinforcementLearningTrader(
        "RLTrader", INITIAL_BALANCE, broker=broker, dry_run=False
    )
    agent.warm_up(broker, cycles=5)

    if not agent.ready:
        # Two distinct reasons warm_up() exits without setting ready=True:
        #
        #   A) USE_RL_AGENT=false or no checkpoint → passive/safe mode.
        #      The loop still runs so health metrics and price polling
        #      continue; step() returns zero deltas so no orders fire.
        #
        #   B) Broker connectivity failure inside the warm_up cycles.
        #      The broker's kill-switch fires; abort cleanly.
        #
        # Distinguish by the RL gate flags, not broker state.
        if not agent.USE_RL_AGENT or not agent._checkpoint_loaded:
            print(
                "[MAIN] RL agent is inactive — running MA crossover strategy.\n"
                "       The MA strategy will begin placing paper trades once it has\n"
                "       accumulated enough price history (20 bars ≈ 20 seconds).\n"
                "       Set USE_RL_AGENT=true in Secrets to switch to the RL actor."
            )
        else:
            print("[MAIN] Warm-up failed (broker/connectivity issue) — aborting.")
            if isinstance(broker, PaperBroker):
                broker.close()
            return
    else:
        print("[MAIN] Warm-up complete — entering live trading loop (Ctrl-C to stop)")

    try:
        run_live_trading_loop(broker, agent)
    finally:
        # Always flush the trade log on exit (KeyboardInterrupt, exception, or normal)
        if isinstance(broker, PaperBroker):
            broker.close()


def main():
    if MODE == "live":
        run_live()
        return

    # Create agents once — RLTrader's Q-table accumulates across episodes
    agents = [
        ValueTrader("ValueTrader", INITIAL_BALANCE),
        MomentumTrader("MomentumTrader", INITIAL_BALANCE),
        RandomTrader("RandomTrader", INITIAL_BALANCE),
        ReinforcementLearningTrader("RLTrader", INITIAL_BALANCE),
    ]
    rl_trader = agents[-1]

    print("Agent latency configuration:")
    for a in agents:
        print(f"  {a.name:<18} latency = {a.latency} step(s)")
    print()

    # Single RiskManager — register_agents() resets per-episode state each run
    risk_manager = RiskManager(
        max_position=10,
        max_notional_per_order=1_000,
        max_drawdown=0.20,
        global_max_drawdown=0.30,
    )

    episode_summaries = []
    qtable_sizes = []
    price_history = []
    regime_history = []

    for episode in range(1, NUM_EPISODES + 1):
        # Reset per-episode agent state; Q-table preserved on RLTrader
        for agent in agents:
            agent.reset_for_new_episode()

        # --- Data source: mode switch ------------------------------------
        if MODE == "sim":
            data_source = SimulatedDataSource(MarketAgent(MARKET_START_PRICE))
            broker = SimulatedBroker(Exchange())
        else:
            raise NotImplementedError(
                f"BOT_MODE={MODE!r} is not a valid simulation mode. "
                "Use BOT_MODE=sim for the multi-agent simulation, or "
                "BOT_MODE=live (default) for the live/paper trading loop."
            )

        # --- Fresh simulation with shared agents and risk manager --------
        sim = Simulation(
            agents=agents,
            market_data_source=data_source,
            broker=broker,
            risk_manager=risk_manager,
        )
        trades, agents, price_history, regime_history = sim.run()

        # Collect stats
        qtable_sizes.append(len(rl_trader.replay_buffer))
        episode_equity = {a.name: round(a.balance + a.unrealized_pnl, 2) for a in agents}
        episode_summaries.append(episode_equity)

        regime_label = sim.initial_regime.upper()
        if regime_history and regime_history[-1] != sim.initial_regime:
            regime_label += f" → {regime_history[-1].upper()}"

        print(f"=== Episode {episode}/{NUM_EPISODES}  [Regime: {regime_label}] ===")
        for a in agents:
            eq = a.balance + a.unrealized_pnl
            tag = (
                f"  (C51 {rl_trader.feature_dim}feat×({rl_trader.num_assets}-asset actor)×{rl_trader.num_atoms}atoms"
                f", buf={len(rl_trader.replay_buffer)})"
                if isinstance(a, ReinforcementLearningTrader) else ""
            )
            print(f"  {a.name:<18} equity: {eq:>10.2f}{tag}")
        print_risk_summary(risk_manager)
        if hasattr(sim, "macro_messages") and sim.macro_messages:
            print(f"  Macro feedback ({len(sim.macro_messages)} alert(s) this episode):")
            for msg in sim.macro_messages[-5:]:
                print(f"    - {msg}")
        print()

    # ---------------------------------------------------------------------------
    # Final cross-episode summary
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print(f"MULTI-EPISODE SUMMARY  ({NUM_EPISODES} episodes)")
    print("=" * 60)
    for name in [a.name for a in agents]:
        equities = [ep[name] for ep in episode_summaries]
        avg = sum(equities) / len(equities)
        best = max(equities)
        worst = min(equities)
        print(f"{name:<18}  avg: {avg:>9.2f}  best: {best:>9.2f}  worst: {worst:>9.2f}")

    print(f"\nRLTrader C51: {rl_trader.feature_dim} features × ({rl_trader.num_assets}-asset actor)"
          f" × {rl_trader.num_atoms} atoms  |  replay buffer: {len(rl_trader.replay_buffer)} transitions")

    print("\nFinal Episode — Performance Detail:")
    for agent in agents:
        total_equity = agent.balance + agent.unrealized_pnl
        print(f"  {agent.name}:")
        print(f"    Realized PnL:   {agent.realized_pnl:.2f}")
        print(f"    Unrealized PnL: {agent.unrealized_pnl:.2f}")
        print(f"    Total Equity:   {total_equity:.2f}")

    # Show a sample of fills from the final episode to verify slippage + fees
    print(f"\nFinal Episode — Trade Sample (first 12 fills of {len(trades)}):")
    print(f"  {'Agent':<18} {'Side':<5} {'Qty':>4}  {'Exec Price':>12}  {'Fee':>8}")
    print(f"  {'-'*18} {'-'*5} {'-'*4}  {'-'*12}  {'-'*8}")
    for name, side, exec_price, qty, fee in trades[:12]:
        print(f"  {name:<18} {side:<5} {qty:>4}  {exec_price:>12.4f}  {fee:>8.4f}")

    print("\nGenerating charts...")
    plot_price_chart(price_history, regime_history)
    plot_equity_curves(agents)
    plot_drawdowns(agents)
    plot_rl_qtable_growth(qtable_sizes)


if __name__ == "__main__":
    main()
