import random

class MarketAgent:
    def __init__(self, start_price, drift=0.0005, volatility=0.02):
        self.price = start_price
        self.drift = drift          # long-term upward/downward bias
        self.volatility = volatility  # typical % movement per step

    def update_price(self):
        # Random percentage move based on volatility
        pct_change = random.gauss(self.drift, self.volatility)

        # Apply percentage change
        self.price *= (1 + pct_change)

        # Rare shock events (1% chance)
        if random.random() < 0.01:
            shock = random.uniform(-0.15, 0.15)  # -15% to +15%
            self.price *= (1 + shock)

        # Prevent price from going to zero
        self.price = max(1, self.price)

        return round(self.price, 2)
