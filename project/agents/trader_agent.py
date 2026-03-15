import random

class TraderAgent:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.position = 0

    def decide(self, market_price):
        # 20% chance to buy if price is low
        if market_price < 100 and random.random() < 0.2:
            return "buy"

        # 20% chance to sell if price is high
        if market_price > 110 and self.position > 0 and random.random() < 0.2:
            return "sell"

        # Otherwise hold
        return "hold"
