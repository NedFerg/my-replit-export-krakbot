class TraderAgent:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.position = 0  # positive = long, negative = short

    def decide(self, market_price):
        # Placeholder decision logic
        # Later we can replace this with real strategy
        if market_price < 100:
            return "buy"
        elif market_price > 110:
            return "sell"
        else:
            return "hold"
