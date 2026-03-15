from simulation.simulation import Simulation

def main():
    sim = Simulation()
    trades = sim.run()

    print(f"Market Regime: {sim.initial_regime.upper()}")
    if sim.market.regime != sim.initial_regime:
        print(f"  (Regime shifted to: {sim.market.regime.upper()} during simulation)")

    print("\nSimulation complete.")
    print("Trades:")
    for trade in trades:
        name, side, price = trade
        print(f"{name} {side} at {price}")

    print("\nPerformance Summary:")
    for agent in sim.agents:
        total_equity = agent.balance + agent.unrealized_pnl
        print(f"{agent.name}:")
        print(f"  Realized PnL: {agent.realized_pnl:.2f}")
        print(f"  Unrealized PnL: {agent.unrealized_pnl:.2f}")
        print(f"  Final Balance: {agent.balance:.2f}")
        print(f"  Total Equity: {total_equity:.2f}\n")

if __name__ == "__main__":
    main()
