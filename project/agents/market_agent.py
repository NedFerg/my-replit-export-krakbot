import random


class MarketAgent:
    def __init__(self, start_price, base_drift=0.0005, base_volatility=0.02):
        self.price = start_price
        self.base_drift = base_drift
        self.base_volatility = base_volatility
        self.drift = base_drift
        self.volatility = base_volatility
        self._crash_triggered = False
        self.regime = self.choose_regime()

    def choose_regime(self):
        regimes = ["bull", "bear", "low_vol", "high_vol", "crash"]
        weights = [0.30, 0.30, 0.20, 0.15, 0.05]
        return random.choices(regimes, weights=weights, k=1)[0]

    def apply_regime_effects(self):
        """Adjust drift and volatility to reflect the current regime."""
        if self.regime == "bull":
            self.drift = self.base_drift + 0.002
            self.volatility = self.base_volatility

        elif self.regime == "bear":
            self.drift = self.base_drift - 0.003
            self.volatility = self.base_volatility

        elif self.regime == "low_vol":
            self.drift = self.base_drift
            self.volatility = self.base_volatility * 0.4

        elif self.regime == "high_vol":
            self.drift = self.base_drift
            self.volatility = self.base_volatility * 2.5

        elif self.regime == "crash":
            # After the shock, crash behaves like a bear market
            self.drift = self.base_drift - 0.003
            self.volatility = self.base_volatility

    def update_price(self):
        # One-time crash shock — fires on the first step in crash regime
        if self.regime == "crash" and not self._crash_triggered:
            self._crash_triggered = True
            shock = random.uniform(-0.30, -0.10)
            self.price *= (1 + shock)
            self.price = max(1, self.price)
            self.regime = "bear"  # revert to bear after the shock

        # Set drift and volatility based on current regime
        self.apply_regime_effects()

        # Gaussian random walk
        pct_change = random.gauss(self.drift, self.volatility)
        self.price *= (1 + pct_change)

        # Rare additional shock events (1% chance)
        if random.random() < 0.01:
            shock = random.uniform(-0.15, 0.15)
            self.price *= (1 + shock)

        self.price = max(1, self.price)
        return round(self.price, 2)
