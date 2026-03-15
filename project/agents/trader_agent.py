import random

class TraderAgent:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.position = 0
        self.last_price = None

    def update_last_price(self, price):
        self.last_price = price

    def decide(self, market_price):
        raise NotImplementedError("Subclasses must implement decide()")


class ValueTrader(TraderAgent):
    def decide(self, price):
        # Buy when price is low
        if price < 95 and random.random() < 0.4:
            return "buy"

        # Sell when price is high
        if price > 110 and self.position > 0 and random.random() < 0.4:
            return "sell"

        return "hold"


class MomentumTrader(TraderAgent):
    def decide(self, price):
        if self.last_price is None:
            return "hold"

        # Buy when price is rising
        if price > self.last_price and random.random() < 0.3:
            return "buy"

        # Sell when price is falling
        if price < self.last_price and self.position > 0 and random.random() < 0.3:
            return "sell"

        return "hold"


class RandomTrader(TraderAgent):
    def decide(self, price):
        r = random.random()
        if r < 0.05:
            return "buy"
        if r < 0.10 and self.position > 0:
            return "sell"
        return "hold"
