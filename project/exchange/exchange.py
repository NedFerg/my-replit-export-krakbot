class MarketState:
    """Snapshot of the order book and market conditions at a point in time."""

    def __init__(self, last_price, best_bid, best_ask, bid_size, ask_size,
                 regime, drift, volatility):
        self.last_price = last_price
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.regime = regime
        self.drift = drift
        self.volatility = volatility

        if best_bid is not None and best_ask is not None:
            self.spread = round(best_ask - best_bid, 4)
            self.mid_price = round((best_bid + best_ask) / 2, 4)
        else:
            self.spread = None
            self.mid_price = last_price


class _MarketMaker:
    """Provides two-sided liquidity as counterparty of last resort."""
    name = "MarketMaker"
    balance = float('inf')
    position = 0
    realized_pnl = 0
    unrealized_pnl = 0


class Order:
    _id_counter = 0

    def __init__(self, agent, side, price, quantity=1):
        Order._id_counter += 1
        self.id = Order._id_counter
        self.agent = agent
        self.side = side
        self.price = price
        self.quantity = quantity


class OrderBook:
    def __init__(self, trade_log):
        self.bids = []  # buy orders: sorted by price desc, then FIFO
        self.asks = []  # sell orders: sorted by price asc, then FIFO
        self.trade_log = trade_log

    def add_order(self, order):
        if order.side == "buy":
            self.bids.append(order)
            self.bids.sort(key=lambda o: (-o.price, o.id))
        elif order.side == "sell":
            self.asks.append(order)
            self.asks.sort(key=lambda o: (o.price, o.id))

    def get_best_bid(self):
        return self.bids[0].price if self.bids else None

    def get_best_ask(self):
        return self.asks[0].price if self.asks else None

    def match(self):
        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            if best_bid.price < best_ask.price:
                break

            if best_bid.agent is best_ask.agent:
                break

            trade_price = best_ask.price
            trade_qty = min(best_bid.quantity, best_ask.quantity)

            buyer = best_bid.agent
            seller = best_ask.agent

            buyer.balance -= trade_price * trade_qty
            buyer.position += trade_qty
            buyer.realized_pnl -= trade_price * trade_qty
            if buyer.name != "MarketMaker":
                self.trade_log.append((buyer.name, "BUY", trade_price))

            seller.balance += trade_price * trade_qty
            seller.position -= trade_qty
            seller.realized_pnl += trade_price * trade_qty
            if seller.name != "MarketMaker":
                self.trade_log.append((seller.name, "SELL", trade_price))

            best_bid.quantity -= trade_qty
            best_ask.quantity -= trade_qty

            if best_bid.quantity == 0:
                self.bids.pop(0)
            if best_ask.quantity == 0:
                self.asks.pop(0)


class Exchange:
    def __init__(self, market_agent):
        self.market = market_agent
        self.trade_log = []
        self.order_book = OrderBook(self.trade_log)
        self._mm = _MarketMaker()

    def get_market_state(self, current_price, regime, drift, volatility) -> MarketState:
        """Build a read-only MarketState snapshot from the current order book."""
        agent_bids = [o for o in self.order_book.bids if o.agent is not self._mm]
        agent_asks = [o for o in self.order_book.asks if o.agent is not self._mm]

        best_bid = agent_bids[0].price if agent_bids else None
        best_ask = agent_asks[0].price if agent_asks else None
        bid_size = agent_bids[0].quantity if agent_bids else 0
        ask_size = agent_asks[0].quantity if agent_asks else 0

        return MarketState(current_price, best_bid, best_ask, bid_size, ask_size,
                           regime, drift, volatility)

    def process_order(self, agent, action, market_price):
        """Submit a limit order and attempt agent-to-agent matching."""
        if action == "buy" and agent.balance >= market_price:
            order = Order(agent, "buy", market_price)
            self.order_book.add_order(order)
            self.order_book.match()

        elif action == "sell" and agent.position > 0:
            order = Order(agent, "sell", market_price)
            self.order_book.add_order(order)
            self.order_book.match()

    def fill_resting_orders(self, price):
        """Fill any unmatched orders using the market maker as counterparty of last resort."""
        mm = self._mm

        self.order_book.bids = [o for o in self.order_book.bids if o.agent is not mm]
        self.order_book.asks = [o for o in self.order_book.asks if o.agent is not mm]

        has_agent_bids = any(o.agent is not mm for o in self.order_book.bids)
        has_agent_asks = any(o.agent is not mm for o in self.order_book.asks)

        if has_agent_bids:
            self.order_book.add_order(Order(mm, "sell", price))
        if has_agent_asks:
            self.order_book.add_order(Order(mm, "buy", price))

        if has_agent_bids or has_agent_asks:
            self.order_book.match()

    def update_market(self):
        return self.market.update_price()
