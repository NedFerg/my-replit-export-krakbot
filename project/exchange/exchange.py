SLIPPAGE_K = 0.0005  # market-impact coefficient per unit vs. available book depth
FEE_RATE = 0.001     # 0.1% taker fee applied to each fill (market maker is exempt)


class Fill:
    """Records one completed execution with realistic price, quantity, and fee."""

    __slots__ = ("agent_name", "side", "price", "quantity", "fee")

    def __init__(self, agent_name, side, price, quantity, fee=0.0):
        self.agent_name = agent_name
        self.side = side
        self.price = price
        self.quantity = quantity
        self.fee = fee

    def __repr__(self):
        return (
            f"Fill({self.agent_name!r}, {self.side!r}, "
            f"price={self.price}, qty={self.quantity}, fee={self.fee:.4f})"
        )


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

    def match(self) -> list:
        """
        Match resting orders by price-time priority.

        Handles partial fills: if the best bid quantity > best ask quantity (or
        vice versa), the residual stays on the book and matching continues with
        the next level.

        Applies a taker fee (FEE_RATE) to every agent fill; the market maker
        is exempt.  Fee is deducted from the agent's balance immediately.

        Returns a list of Fill objects for every execution that occurred.
        """
        fills = []

        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            if best_bid.price < best_ask.price:
                break
            if best_bid.agent is best_ask.agent:
                break

            trade_price = best_ask.price
            trade_qty = min(best_bid.quantity, best_ask.quantity)
            cost = trade_price * trade_qty

            buyer = best_bid.agent
            seller = best_ask.agent

            # --- Apply cash changes at actual execution price ---
            # Position is only mutated for classical (non-RL) agents.
            # The RL agent manages its own exposure float via the broker.
            buyer.balance -= cost
            if not getattr(buyer, 'IS_RL_AGENT', False):
                buyer.position += trade_qty
            buyer.realized_pnl -= cost

            seller.balance += cost
            if not getattr(seller, 'IS_RL_AGENT', False):
                seller.position -= trade_qty
            seller.realized_pnl += cost

            # --- Taker fees (market maker is exempt) -----------------------
            buyer_fee = 0.0
            seller_fee = 0.0

            if buyer.name != "MarketMaker":
                buyer_fee = round(trade_price * trade_qty * FEE_RATE, 6)
                buyer.balance -= buyer_fee
                fill = Fill(buyer.name, "buy", trade_price, trade_qty, buyer_fee)
                fills.append(fill)
                self.trade_log.append(
                    (buyer.name, "BUY", trade_price, trade_qty, buyer_fee)
                )

            if seller.name != "MarketMaker":
                seller_fee = round(trade_price * trade_qty * FEE_RATE, 6)
                seller.balance -= seller_fee
                fill = Fill(seller.name, "sell", trade_price, trade_qty, seller_fee)
                fills.append(fill)
                self.trade_log.append(
                    (seller.name, "SELL", trade_price, trade_qty, seller_fee)
                )

            # --- Reduce remaining quantities — enables partial fills -------
            best_bid.quantity -= trade_qty
            best_ask.quantity -= trade_qty

            if best_bid.quantity == 0:
                self.bids.pop(0)
            if best_ask.quantity == 0:
                self.asks.pop(0)

        return fills


class Exchange:
    def __init__(self, market_agent=None):
        self.market = market_agent  # optional: only needed if calling update_market()
        self.trade_log = []
        self.order_book = OrderBook(self.trade_log)
        self._mm = _MarketMaker()

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------

    def _slippage_price(self, base_price: float, side: str, quantity: int) -> float:
        """
        Compute an execution price that reflects market impact.

        Buys execute slightly above base_price; sells slightly below.
        Impact shrinks as available book depth grows.
        """
        if side == "buy":
            available = sum(o.quantity for o in self.order_book.asks)
        else:
            available = sum(o.quantity for o in self.order_book.bids)

        depth = max(available, 1)  # floor at 1 to avoid division by zero
        impact = SLIPPAGE_K * quantity / depth
        direction = 1.0 if side == "buy" else -1.0
        return round(base_price * (1.0 + impact * direction), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def process_order(self, agent, action, market_price) -> list:
        """
        Submit a limit order at a slippage-adjusted price and match immediately.

        Returns a (possibly empty) list of Fill objects produced by matching.
        Partial fills are supported: if the book cannot absorb the full quantity,
        the residual rests in the book until fill_resting_orders() is called.
        """
        exec_price = self._slippage_price(market_price, action, quantity=1)

        if action == "buy" and agent.balance >= exec_price:
            order = Order(agent, "buy", exec_price)
            self.order_book.add_order(order)
            return self.order_book.match()

        elif action == "sell" and (getattr(agent, 'IS_RL_AGENT', False) or agent.position > 0):
            order = Order(agent, "sell", exec_price)
            self.order_book.add_order(order)
            return self.order_book.match()

        return []

    def fill_resting_orders(self, price) -> list:
        """
        Fill any unmatched resting agent orders using the market maker as
        counterparty of last resort.  Returns fills produced by the final match.
        """
        mm = self._mm

        # Remove any leftover market-maker orders from the previous step
        self.order_book.bids = [o for o in self.order_book.bids if o.agent is not mm]
        self.order_book.asks = [o for o in self.order_book.asks if o.agent is not mm]

        has_agent_bids = any(o.agent is not mm for o in self.order_book.bids)
        has_agent_asks = any(o.agent is not mm for o in self.order_book.asks)

        if has_agent_bids:
            self.order_book.add_order(Order(mm, "sell", price))
        if has_agent_asks:
            self.order_book.add_order(Order(mm, "buy", price))

        if has_agent_bids or has_agent_asks:
            return self.order_book.match()

        return []

    def update_market(self):
        return self.market.update_price()
