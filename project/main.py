from simulation.simulation import Simulation

def main():
    sim = Simulation()
    trades = sim.run()

    print("Simulation complete.")
    print("Trades:")
    for trade in trades:
        name, side, price = trade
        print(f"{name} {side} at {price}")

if __name__ == "__main__":
    main()
