class MarketAgent:
    def __init__(self, start_price):
        self.price = start_price

    def update_price(self):
        # Simple placeholder logic for now
        # Later we can add volatility, trends, randomness, etc.
        self.price += 1  # price drifts upward for demonstration
        return self.price
