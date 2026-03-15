import random


class TraderAgent:
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
        self.position = 0
        self.last_price = None  # tracks last mid_price for momentum comparison
        self.realized_pnl = 0
        self.unrealized_pnl = 0

    def update_last_price(self, price):
        self.last_price = price

    def decide(self, market_state):
        raise NotImplementedError("Subclasses must implement decide()")

    def update_unrealized_pnl(self, market_price):
        self.unrealized_pnl = self.position * market_price


class ValueTrader(TraderAgent):
    def decide(self, market_state):
        price = market_state.mid_price

        # Skip trading when the spread is too wide (costly to cross)
        if market_state.spread is not None and market_state.spread > 5:
            return "hold"

        if price < 95 and random.random() < 0.4:
            return "buy"

        if price > 110 and self.position > 0 and random.random() < 0.4:
            return "sell"

        return "hold"


class MomentumTrader(TraderAgent):
    def decide(self, market_state):
        price = market_state.mid_price

        if self.last_price is None:
            return "hold"

        # Increase activity when order book shows strong buy-side imbalance
        buy_bias = market_state.bid_size > market_state.ask_size

        # Buy when price is rising (boost probability on imbalance)
        if price > self.last_price:
            threshold = 0.4 if buy_bias else 0.3
            if random.random() < threshold:
                return "buy"

        # Sell when price is falling and we hold a position
        if price < self.last_price and self.position > 0:
            if random.random() < 0.3:
                return "sell"

        return "hold"


class RandomTrader(TraderAgent):
    def decide(self, market_state):
        # Trade more frequently when spread is tight (cheap to cross)
        if market_state.spread is not None and market_state.spread < 1:
            buy_threshold = 0.08
            sell_threshold = 0.16
        else:
            buy_threshold = 0.05
            sell_threshold = 0.10

        r = random.random()
        if r < buy_threshold:
            return "buy"
        if r < sell_threshold and self.position > 0:
            return "sell"
        return "hold"
