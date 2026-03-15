import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.simulation import Simulation
from agents.rl_agent import ReinforcementLearningTrader


REGIME_COLORS = {
    "bull":     "green",
    "bear":     "red",
    "high_vol": "orange",
    "low_vol":  "blue",
    "crash":    "purple",
}


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
    ax.set_title("Price History with Regime Shading")
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
    ax.set_title("Agent Equity Curves")
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
    ax.set_title("Agent Drawdowns")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("drawdowns.png", dpi=150)
    plt.close()
    print("Saved: drawdowns.png")


def main():
    sim = Simulation()
    trades, agents, price_history, regime_history = sim.run()

    print(f"Market Regime: {sim.initial_regime.upper()}")
    if sim.market.regime != sim.initial_regime:
        print(f"  (Regime shifted to: {sim.market.regime.upper()} during simulation)")

    print("\nSimulation complete.")
    print("Trades:")
    for trade in trades:
        name, side, price = trade
        print(f"{name} {side} at {price}")

    print("\nPerformance Summary:")
    for agent in agents:
        total_equity = agent.balance + agent.unrealized_pnl
        print(f"{agent.name}:")
        print(f"  Realized PnL:   {agent.realized_pnl:.2f}")
        print(f"  Unrealized PnL: {agent.unrealized_pnl:.2f}")
        print(f"  Final Balance:  {agent.balance:.2f}")
        print(f"  Total Equity:   {total_equity:.2f}")
        if isinstance(agent, ReinforcementLearningTrader):
            print(f"  Q-table states: {len(agent.q_table)}")
        print()

    print("Generating charts...")
    plot_price_chart(price_history, regime_history)
    plot_equity_curves(agents)
    plot_drawdowns(agents)


if __name__ == "__main__":
    main()
