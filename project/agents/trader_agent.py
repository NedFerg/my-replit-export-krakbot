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
