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

# ---------------------------------------------------------------------------
# Mode switch — "sim" runs the local simulation; "live" starts the Kraken
# live trading loop (requires KRAKEN_API_KEY / KRAKEN_API_SECRET secrets).
# ---------------------------------------------------------------------------
MODE = "live"

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

    total_blocked = sum(rejected.values())
    if total_blocked == 0 and not locked and not gks:
        print("  [Risk] No limits hit this episode.")
        return

    parts = [f"{name}={n}" for name, n in rejected.items() if n > 0]
    print(f"  [Risk] Orders blocked: {', '.join(parts) if parts else 'none'}")
    if locked:
        print(f"  [Risk] Drawdown-locked: {', '.join(locked)}")
    if gks:
        print("  [Risk] *** GLOBAL KILL-SWITCH triggered ***")


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
    """
    import os as _os
    use_paper = _os.environ.get("USE_PAPER_BROKER", "true").strip().lower() \
                not in ("false", "0", "no")

    print("[MAIN] Starting live trading loop...")

    if use_paper:
        broker = PaperBroker(initial_cash=INITIAL_BALANCE)
    else:
        broker = LiveBroker(dry_run=False)

    print(f"[MAIN] Broker mode: {_broker_mode_label(broker)}")

    # Futures wallet — paper-simulated when ENABLE_FUTURES=False (default)
    if broker.futures_paper_mode:
        print("[MAIN] Futures overlay: PAPER mode — simulating collateral from spot equity.")
    else:
        print("[MAIN] Futures overlay: LIVE mode — fetching real collateral from futures.kraken.com.")
    broker.fetch_futures_wallet()

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
                "       accumulated enough price history (100 bars ≈ 50 minutes).\n"
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
                f"MODE={MODE!r} is not implemented. "
                "Add a MarketDataSource and Broker subclass for live or paper trading."
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
