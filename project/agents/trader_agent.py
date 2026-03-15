import random

class TraderAgent:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.position = 0  # number of units held

    def decide(self, market_price):
        # More active logic:
        # - Buy when price is relatively low
        # - Sell when price is relatively high and we hold something

        # 30% chance to buy when price is below 100
        if market_price < 100 and random.random() < 0.3:
            return "buy"

        # 30% chance to sell when price is above 105 and we have a position
        if market_price > 105 and self.position > 0 and random.random() < 0.3:
            return "sell"

        # Otherwise hold
        return "hold"
