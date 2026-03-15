import random

class MarketAgent:
    def __init__(self, start_price):
        self.price = start_price

    def update_price(self):
        # Random walk: price moves up or down by 1–3
        change = random.randint(-3, 3)
        self.price = max(1, self.price + change)
        return self.price
