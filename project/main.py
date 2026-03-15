import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.simulation import Simulation
from agents.trader_agent import ValueTrader, MomentumTrader, RandomTrader
from agents.rl_agent import ReinforcementLearningTrader
from config.config import INITIAL_BALANCE

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
    """Shade chart background by market regime."""
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
        for r in REGIME_COLORS
        if r in regime_history
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Create agents once — RLTrader's Q-table will accumulate across episodes
    agents = [
        ValueTrader("ValueTrader", INITIAL_BALANCE),
        MomentumTrader("MomentumTrader", INITIAL_BALANCE),
        RandomTrader("RandomTrader", INITIAL_BALANCE),
        ReinforcementLearningTrader("RLTrader", INITIAL_BALANCE),
    ]
    rl_trader = agents[-1]

    episode_summaries = []  # list of {name: final_equity} per episode
    qtable_sizes = []       # Q-table size after each episode

    price_history = []
    regime_history = []

    for episode in range(1, NUM_EPISODES + 1):
        # Reset per-episode state; Q-table is preserved on RLTrader
        for agent in agents:
            agent.reset_for_new_episode()

        # Fresh market + exchange, same agents
        sim = Simulation(agents=agents)
        trades, agents, price_history, regime_history = sim.run()

        # Per-episode stats
        qtable_sizes.append(len(rl_trader.q_table))
        episode_equity = {a.name: round(a.balance + a.unrealized_pnl, 2) for a in agents}
        episode_summaries.append(episode_equity)

        regime_label = sim.initial_regime.upper()
        if sim.market.regime != sim.initial_regime:
            regime_label += f" → {sim.market.regime.upper()}"

        print(f"=== Episode {episode}/{NUM_EPISODES}  [Regime: {regime_label}] ===")
        for a in agents:
            eq = a.balance + a.unrealized_pnl
            tag = f"  (Q-table: {len(rl_trader.q_table)} states)" if isinstance(a, ReinforcementLearningTrader) else ""
            print(f"  {a.name:<18} equity: {eq:>10.2f}{tag}")
        print()

    # ---------------------------------------------------------------------------
    # Final summary across all episodes
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

    print(f"\nRLTrader final Q-table: {len(rl_trader.q_table)} unique states")

    # Final episode performance detail
    print("\nFinal Episode — Performance Detail:")
    for agent in agents:
        total_equity = agent.balance + agent.unrealized_pnl
        print(f"  {agent.name}:")
        print(f"    Realized PnL:   {agent.realized_pnl:.2f}")
        print(f"    Unrealized PnL: {agent.unrealized_pnl:.2f}")
        print(f"    Total Equity:   {total_equity:.2f}")

    # ---------------------------------------------------------------------------
    # Charts (final episode data + RL Q-table growth)
    # ---------------------------------------------------------------------------
    print("\nGenerating charts...")
    plot_price_chart(price_history, regime_history)
    plot_equity_curves(agents)
    plot_drawdowns(agents)
    plot_rl_qtable_growth(qtable_sizes)


if __name__ == "__main__":
    main()
