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
        regime = market_state.regime

        # Skip when spread is too wide — costly to cross
        if market_state.spread is not None and market_state.spread > 5:
            return "hold"

        # Adjust thresholds and probabilities based on regime
        if regime == "bull":
            buy_thresh, buy_prob = 97, 0.45
            sell_thresh, sell_prob = 110, 0.30
        elif regime == "bear":
            buy_thresh, buy_prob = 90, 0.30
            sell_thresh, sell_prob = 105, 0.50
        elif regime == "high_vol":
            # Be more cautious — widen thresholds
            buy_thresh, buy_prob = 85, 0.35
            sell_thresh, sell_prob = 120, 0.35
        elif regime == "low_vol":
            # Tighter market — trade more actively
            buy_thresh, buy_prob = 97, 0.50
            sell_thresh, sell_prob = 105, 0.45
        else:
            buy_thresh, buy_prob = 95, 0.40
            sell_thresh, sell_prob = 110, 0.40

        if price < buy_thresh and random.random() < buy_prob:
            return "buy"
        if price > sell_thresh and self.position > 0 and random.random() < sell_prob:
            return "sell"
        return "hold"


class MomentumTrader(TraderAgent):
    def decide(self, market_state):
        price = market_state.mid_price
        regime = market_state.regime

        if self.last_price is None:
            return "hold"

        buy_bias = market_state.bid_size > market_state.ask_size

        # Regime-adjusted momentum sensitivity
        if regime == "bull":
            up_prob = 0.45 if buy_bias else 0.35
            down_prob = 0.20
        elif regime == "bear":
            # More aggressive selling on downward momentum
            up_prob = 0.20
            down_prob = 0.45
        elif regime == "high_vol":
            # Faster reactions in volatile markets
            up_prob = 0.40 if buy_bias else 0.35
            down_prob = 0.40
        elif regime == "low_vol":
            # Quieter market — reduce sensitivity to avoid noise
            up_prob = 0.20
            down_prob = 0.15
        else:
            up_prob = 0.40 if buy_bias else 0.30
            down_prob = 0.30

        if price > self.last_price and random.random() < up_prob:
            return "buy"
        if price < self.last_price and self.position > 0 and random.random() < down_prob:
            return "sell"
        return "hold"


class RandomTrader(TraderAgent):
    def decide(self, market_state):
        regime = market_state.regime

        # Base probabilities by regime
        if regime == "high_vol":
            buy_p, sell_p = 0.03, 0.06  # trade less in choppy markets
        elif regime == "low_vol":
            buy_p, sell_p = 0.08, 0.16  # trade more in quiet markets
        else:
            buy_p, sell_p = 0.05, 0.10

        # Boost when spread is tight — cheap to cross
        if market_state.spread is not None and market_state.spread < 1:
            buy_p = min(buy_p * 1.5, 0.20)
            sell_p = min(sell_p * 1.5, 0.30)

        r = random.random()
        if r < buy_p:
            return "buy"
        if r < buy_p + sell_p and self.position > 0:
            return "sell"
        return "hold"
