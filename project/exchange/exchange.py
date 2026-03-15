class Exchange:
    def __init__(self, market_agent):
        self.market = market_agent
        self.trade_log = []

    def process_order(self, agent, action):
        price = self.market.price

        if action == "buy":
            if agent.balance >= price:
                agent.balance -= price
                agent.position += 1
                agent.realized_pnl -= price  # cost of buying
                self.trade_log.append((agent.name, "BUY", price))
        elif action == "sell":
            if agent.position > 0:
                agent.balance += price
                agent.position -= 1
                agent.realized_pnl += price  # profit from selling
                self.trade_log.append((agent.name, "SELL", price))

    def update_market(self):
        return self.market.update_price()
