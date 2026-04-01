"""
tests/test_alpaca_broker.py
----------------------------
Unit tests for the AlpacaBroker class (project/broker/alpaca_broker.py).

All tests run without a real Alpaca API connection — the _api attribute
is patched with a MagicMock that returns pre-defined account / position /
trade responses.

Test coverage
-------------
  • Symbol resolution (crypto, ETF, unknown)
  • Price feed (crypto + ETF, market-hours gate)
  • Account equity and balances
  • Position sync (internal dict update)
  • Order placement (dry-run, live, PDT block, safety block)
  • PDT compliance checks (margin account, cash account, blocked/allowed)
  • Safety checks (trade rate, daily loss, per-asset notional cap)
  • ETF overlay (regime-driven orders routed to Alpaca ETF symbols)
  • Credential validation
  • Asset maps (completeness)
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Allow importing from project/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from broker.alpaca_broker import (
    AlpacaBroker,
    ALPACA_CRYPTO_PAIRS,
    ALPACA_ETF_PAIRS,
    create_alpaca_broker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker(dry_run: bool = True) -> AlpacaBroker:
    """Create an AlpacaBroker with dummy credentials and no real HTTP calls."""
    with patch.dict(os.environ, {
        "ALPACA_API_KEY":    "test_key",
        "ALPACA_API_SECRET": "test_secret",
        "ALPACA_BASE_URL":   "https://paper-api.alpaca.markets",
    }):
        broker = AlpacaBroker(dry_run=dry_run)
    # Replace the real API client with a MagicMock so no HTTP calls are made
    broker._api = MagicMock()
    return broker


def _mock_account(
    equity: float = 50_000.0,
    cash: float = 20_000.0,
    pattern_day_trader: bool = False,
    daytrade_count: int = 0,
    account_type: str = "MARGIN",
) -> MagicMock:
    acct = MagicMock()
    acct.equity             = equity
    acct.cash               = cash
    acct.buying_power       = cash * 2
    acct.portfolio_value    = equity
    acct.pattern_day_trader = pattern_day_trader
    acct.daytrade_count     = daytrade_count
    acct.account_type       = account_type
    return acct


def _mock_trade(price: float) -> MagicMock:
    t = MagicMock()
    t.price = price
    return t


def _mock_order(
    order_id: str = "test-order-id",
    status:   str = "accepted",
) -> MagicMock:
    o = MagicMock()
    o.id     = order_id
    o.status = status
    return o


# ---------------------------------------------------------------------------
# Tests: asset maps
# ---------------------------------------------------------------------------

class TestAssetMaps(unittest.TestCase):
    """Verify the module-level asset mapping constants are correct."""

    def test_all_crypto_assets_present(self):
        required = {"BTC", "ETH", "SOL", "XRP", "LINK", "AVAX", "HBAR", "XLM"}
        for asset in required:
            self.assertIn(asset, ALPACA_CRYPTO_PAIRS,
                          f"Missing crypto mapping for {asset}")

    def test_crypto_symbols_use_slash_notation(self):
        """All Alpaca crypto symbols must use 'BASE/USD' format."""
        for internal, alpaca_sym in ALPACA_CRYPTO_PAIRS.items():
            self.assertIn("/", alpaca_sym,
                          f"{internal} → {alpaca_sym!r} missing '/'")
            self.assertTrue(alpaca_sym.endswith("/USD"),
                            f"{internal} → {alpaca_sym!r} must end with /USD")

    def test_all_etf_assets_present(self):
        required = {"ETHU", "SLON", "XXRP", "ETHD", "SETH"}
        for asset in required:
            self.assertIn(asset, ALPACA_ETF_PAIRS,
                          f"Missing ETF mapping for {asset}")

    def test_short_etfs_map_to_inverse_etfs(self):
        """ETHD and SETH must map to inverse/short equity ETFs."""
        # SQQQ = 3× Short NASDAQ; SH = 1× Short S&P500
        self.assertIn(ALPACA_ETF_PAIRS["ETHD"], {"SQQQ", "SPXS", "PSQ", "SPDN"})
        self.assertIn(ALPACA_ETF_PAIRS["SETH"], {"SH", "SPDN", "SPXS"})

    def test_long_etfs_map_to_leveraged_long_etfs(self):
        self.assertIn(ALPACA_ETF_PAIRS["ETHU"], {"QLD", "SSO", "TQQQ", "UPRO"})
        self.assertIn(ALPACA_ETF_PAIRS["SLON"], {"TQQQ", "UPRO", "QLD", "SSO"})

    def test_etf_values_are_plain_tickers(self):
        """ETF Alpaca symbols must NOT contain '/' (they are stock tickers)."""
        for internal, alpaca_sym in ALPACA_ETF_PAIRS.items():
            self.assertNotIn("/", alpaca_sym,
                             f"{internal} → {alpaca_sym!r} must not contain '/'")


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):

    def test_dry_run_forced_without_credentials(self):
        """No credentials → dry_run is forced to True regardless of param."""
        with patch.dict(os.environ, {
            "ALPACA_API_KEY":    "",
            "ALPACA_API_SECRET": "",
        }):
            broker = AlpacaBroker(dry_run=False)
        self.assertTrue(broker.dry_run)

    def test_dry_run_true_default(self):
        broker = _make_broker()
        self.assertTrue(broker.dry_run)

    def test_all_asset_maps_populated(self):
        broker = _make_broker()
        self.assertTrue(broker.crypto_pairs)
        self.assertTrue(broker.etf_pairs)
        # Combined _all_pairs must include both
        for k in broker.crypto_pairs:
            self.assertIn(k, broker._all_pairs)
        for k in broker.etf_pairs:
            self.assertIn(k, broker._all_pairs)

    def test_kill_switch_starts_false(self):
        broker = _make_broker()
        self.assertFalse(broker.kill_switch)

    def test_etf_positions_initialised_to_zero(self):
        broker = _make_broker()
        for etf in ["ETHU", "SLON", "XXRP", "ETHD", "SETH"]:
            self.assertIn(etf, broker.etf_positions)
            self.assertEqual(broker.etf_positions[etf], 0.0)


# ---------------------------------------------------------------------------
# Tests: symbol resolution
# ---------------------------------------------------------------------------

class TestSymbolResolution(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()

    def test_crypto_resolves_to_alpaca_symbol(self):
        self.assertEqual(self.broker._resolve_symbol("BTC"),  "BTC/USD")
        self.assertEqual(self.broker._resolve_symbol("ETH"),  "ETH/USD")
        self.assertEqual(self.broker._resolve_symbol("SOL"),  "SOL/USD")
        self.assertEqual(self.broker._resolve_symbol("XRP"),  "XRP/USD")
        self.assertEqual(self.broker._resolve_symbol("LINK"), "LINK/USD")
        self.assertEqual(self.broker._resolve_symbol("AVAX"), "AVAX/USD")
        self.assertEqual(self.broker._resolve_symbol("HBAR"), "HBAR/USD")
        self.assertEqual(self.broker._resolve_symbol("XLM"),  "XLM/USD")

    def test_etf_resolves_to_alpaca_ticker(self):
        self.assertEqual(self.broker._resolve_symbol("ETHD"), "SQQQ")
        self.assertEqual(self.broker._resolve_symbol("SETH"), "SH")
        self.assertEqual(self.broker._resolve_symbol("ETHU"), "QLD")
        self.assertEqual(self.broker._resolve_symbol("SLON"), "TQQQ")
        self.assertEqual(self.broker._resolve_symbol("XXRP"), "SSO")

    def test_is_crypto_true_for_slash_symbols(self):
        self.assertTrue(self.broker._is_crypto("BTC/USD"))
        self.assertTrue(self.broker._is_crypto("ETH/USD"))

    def test_is_crypto_false_for_stock_tickers(self):
        self.assertFalse(self.broker._is_crypto("QLD"))
        self.assertFalse(self.broker._is_crypto("SQQQ"))


# ---------------------------------------------------------------------------
# Tests: account equity
# ---------------------------------------------------------------------------

class TestAccountEquity(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()

    def test_get_account_equity_from_api(self):
        self.broker._api.get_account.return_value = _mock_account(equity=12345.67)
        equity = self.broker.get_account_equity()
        self.assertAlmostEqual(equity, 12345.67)

    def test_compute_total_equity_uses_cache_first(self):
        """compute_total_equity should use cached value without API call."""
        self.broker.live_balances = {"equity": 9999.0}
        equity = self.broker.compute_total_equity()
        self.assertAlmostEqual(equity, 9999.0)
        self.broker._api.get_account.assert_not_called()

    def test_get_account_equity_falls_back_on_error(self):
        self.broker._api.get_account.side_effect = Exception("network error")
        self.broker.live_balances = {"equity": 500.0}
        equity = self.broker.get_account_equity()
        self.assertAlmostEqual(equity, 500.0)


# ---------------------------------------------------------------------------
# Tests: fetch_live_balances
# ---------------------------------------------------------------------------

class TestFetchLiveBalances(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()

    def test_populates_live_balances(self):
        self.broker._api.get_account.return_value = _mock_account(
            equity=50_000.0,
            pattern_day_trader=False,
            daytrade_count=1,
            account_type="MARGIN",
        )
        balances = self.broker.fetch_live_balances()
        self.assertIsNotNone(balances)
        self.assertAlmostEqual(balances["equity"], 50_000.0)
        self.assertFalse(balances["pattern_day_trader"])
        self.assertEqual(balances["daytrade_count"], 1)
        self.assertEqual(balances["account_type"], "MARGIN")

    def test_returns_none_on_api_error(self):
        self.broker._api.get_account.side_effect = Exception("auth error")
        balances = self.broker.fetch_live_balances()
        self.assertIsNone(balances)


# ---------------------------------------------------------------------------
# Tests: PDT compliance
# ---------------------------------------------------------------------------

class TestPDTCompliance(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()

    def test_allow_when_no_balance_data(self):
        """Permissive when no account data fetched yet."""
        self.broker.live_balances = {}
        self.assertTrue(self.broker.check_pdt_compliance("SQQQ"))

    def test_allow_cash_account(self):
        self.broker.live_balances = {
            "equity": 5_000.0, "pattern_day_trader": False,
            "daytrade_count": 5, "account_type": "CASH",
        }
        self.assertTrue(self.broker.check_pdt_compliance("SQQQ"))

    def test_allow_margin_account_high_equity(self):
        """Margin account ≥ $25k → always allowed."""
        self.broker.live_balances = {
            "equity": 30_000.0, "pattern_day_trader": False,
            "daytrade_count": 3, "account_type": "MARGIN",
        }
        self.assertTrue(self.broker.check_pdt_compliance("SQQQ"))

    def test_block_pdt_flagged_account_low_equity(self):
        """Margin account already flagged as PDT + equity < $25k → block."""
        self.broker.live_balances = {
            "equity": 10_000.0, "pattern_day_trader": True,
            "daytrade_count": 4, "account_type": "MARGIN",
        }
        self.assertFalse(self.broker.check_pdt_compliance("SQQQ"))

    def test_block_when_three_day_trades_low_equity(self):
        """Margin account at 3 day trades, equity < $25k → block (next would trigger PDT)."""
        self.broker.live_balances = {
            "equity": 8_000.0, "pattern_day_trader": False,
            "daytrade_count": 3, "account_type": "MARGIN",
        }
        self.assertFalse(self.broker.check_pdt_compliance("SQQQ"))

    def test_allow_two_day_trades_low_equity(self):
        """2 day trades in rolling window + low equity → still allowed (not at limit yet)."""
        self.broker.live_balances = {
            "equity": 15_000.0, "pattern_day_trader": False,
            "daytrade_count": 2, "account_type": "MARGIN",
        }
        self.assertTrue(self.broker.check_pdt_compliance("SQQQ"))


# ---------------------------------------------------------------------------
# Tests: place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker(dry_run=True)

    def test_dry_run_returns_status_dry_run(self):
        result = self.broker.place_order("BTC", qty=0.001, side="buy")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["symbol"], "BTC/USD")
        self.broker._api.submit_order.assert_not_called()

    def test_dry_run_etf_order(self):
        result = self.broker.place_order("ETHD", qty=5.0, side="buy", order_type="market")
        self.assertIsNotNone(result)
        self.assertEqual(result["symbol"], "SQQQ")
        self.assertEqual(result["side"], "buy")

    def test_live_order_submitted_to_api(self):
        self.broker.dry_run = False
        self.broker._api.submit_order.return_value = _mock_order(
            order_id="abc-123", status="accepted"
        )
        # Ensure PDT check passes
        self.broker.live_balances = {
            "equity": 50_000.0, "pattern_day_trader": False,
            "daytrade_count": 0, "account_type": "MARGIN",
        }
        result = self.broker.place_order("QLD", qty=3.0, side="buy")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "abc-123")
        self.broker._api.submit_order.assert_called_once()
        call_kwargs = self.broker._api.submit_order.call_args[1]
        self.assertEqual(call_kwargs["symbol"], "QLD")
        self.assertEqual(call_kwargs["qty"],    3.0)
        self.assertEqual(call_kwargs["side"],   "buy")

    def test_zero_qty_returns_none(self):
        result = self.broker.place_order("BTC", qty=0.0, side="buy")
        self.assertIsNone(result)

    def test_invalid_side_returns_none(self):
        result = self.broker.place_order("BTC", qty=0.001, side="hold")
        self.assertIsNone(result)

    def test_kill_switch_blocks_order(self):
        self.broker.kill_switch = True
        result = self.broker.place_order("BTC", qty=0.001, side="buy")
        self.assertIsNone(result)

    def test_pdt_block_prevents_live_etf_order(self):
        self.broker.dry_run = False
        self.broker.live_balances = {
            "equity": 8_000.0, "pattern_day_trader": True,
            "daytrade_count": 4, "account_type": "MARGIN",
        }
        result = self.broker.place_order("ETHD", qty=5.0, side="buy")
        self.assertIsNone(result)
        self.broker._api.submit_order.assert_not_called()

    def test_limit_order_includes_limit_price(self):
        self.broker.dry_run = False
        self.broker._api.submit_order.return_value = _mock_order()
        self.broker.live_balances = {
            "equity": 50_000.0, "pattern_day_trader": False,
            "daytrade_count": 0, "account_type": "MARGIN",
        }
        self.broker.place_order("SQQQ", qty=2.0, side="buy",
                                order_type="limit", price=35.50)
        call_kwargs = self.broker._api.submit_order.call_args[1]
        self.assertEqual(call_kwargs["type"], "limit")
        self.assertIn("limit_price", call_kwargs)
        self.assertEqual(call_kwargs["limit_price"], "35.5")

    def test_limit_order_missing_price_downgrades_to_market(self):
        self.broker.dry_run = False
        self.broker._api.submit_order.return_value = _mock_order()
        self.broker.live_balances = {
            "equity": 50_000.0, "pattern_day_trader": False,
            "daytrade_count": 0, "account_type": "MARGIN",
        }
        self.broker.place_order("SQQQ", qty=2.0, side="buy",
                                order_type="limit", price=None)
        call_kwargs = self.broker._api.submit_order.call_args[1]
        self.assertEqual(call_kwargs["type"], "market")

    def test_soft_error_returns_none_no_kill_switch(self):
        self.broker.dry_run = False
        self.broker._api.submit_order.side_effect = Exception(
            "insufficient buying power"
        )
        self.broker.live_balances = {
            "equity": 50_000.0, "pattern_day_trader": False,
            "daytrade_count": 0, "account_type": "MARGIN",
        }
        result = self.broker.place_order("SQQQ", qty=2.0, side="buy")
        self.assertIsNone(result)
        self.assertFalse(self.broker.kill_switch)   # soft error — no kill switch

    def test_hard_error_triggers_kill_switch(self):
        self.broker.dry_run = False
        self.broker._api.submit_order.side_effect = Exception(
            "unexpected server error"
        )
        self.broker.live_balances = {
            "equity": 50_000.0, "pattern_day_trader": False,
            "daytrade_count": 0, "account_type": "MARGIN",
        }
        self.broker.place_order("SQQQ", qty=2.0, side="buy")
        self.assertTrue(self.broker.kill_switch)


# ---------------------------------------------------------------------------
# Tests: safety checks
# ---------------------------------------------------------------------------

class TestSafetyChecks(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()
        # Give broker a known equity so notional caps are sensible
        self.broker.live_balances = {"equity": 10_000.0}
        self.broker._starting_equity    = None
        self.broker._max_daily_loss_usd = None

    def test_kill_switch_blocks_pre_trade(self):
        self.broker.kill_switch = True
        self.assertFalse(self.broker._pre_trade_safety("BTC", 50_000.0, 100.0))

    def test_invalid_price_blocks_trade(self):
        self.assertFalse(self.broker._pre_trade_safety("BTC", 0.0, 100.0))
        self.assertFalse(self.broker._pre_trade_safety("BTC", -1.0, 100.0))

    def test_below_min_notional_blocks_trade(self):
        # _MIN_NOTIONAL_USD defaults to 1.0
        self.assertFalse(self.broker._pre_trade_safety("BTC", 50_000.0, 0.5))

    def test_per_asset_cap_blocks_oversized_trade(self):
        self.broker.max_notional_per_asset = 100.0
        self.assertFalse(self.broker._pre_trade_safety("BTC", 50_000.0, 101.0))

    def test_valid_trade_passes_all_checks(self):
        self.broker._api.get_account.return_value = _mock_account(equity=10_000.0)
        self.broker.max_notional_per_asset = 1_000.0
        result = self.broker._pre_trade_safety("BTC", 50_000.0, 50.0)
        self.assertTrue(result)

    def test_daily_loss_cap_triggers_kill_switch(self):
        self.broker._starting_equity    = 10_000.0
        self.broker._max_daily_loss_usd = 1_000.0
        # Simulate large loss: equity drops 20% below start
        self.broker._api.get_account.return_value = _mock_account(equity=8_500.0)
        self.broker.live_balances = {"equity": 8_500.0}
        result = self.broker._check_daily_loss()
        self.assertFalse(result)
        self.assertTrue(self.broker.kill_switch)

    def test_trade_rate_limit_triggers_kill_switch(self):
        # Fill the window with max_trades_per_hour entries
        now = time.time()
        self.broker._trade_count_window = [now - 10] * self.broker.max_trades_per_hour
        result = self.broker._check_trade_rate()
        self.assertFalse(result)
        self.assertTrue(self.broker.kill_switch)


# ---------------------------------------------------------------------------
# Tests: price feed
# ---------------------------------------------------------------------------

class TestPriceFeed(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()

    def test_fetch_crypto_prices(self):
        trades = {
            "BTC/USD":  _mock_trade(60_000.0),
            "ETH/USD":  _mock_trade(3_000.0),
            "SOL/USD":  _mock_trade(150.0),
            "XRP/USD":  _mock_trade(1.50),
            "LINK/USD": _mock_trade(15.0),
            "AVAX/USD": _mock_trade(35.0),
            "HBAR/USD": _mock_trade(0.07),
            "XLM/USD":  _mock_trade(0.12),
        }
        self.broker._api.get_latest_crypto_trades.return_value = trades
        # Market closed → no ETF prices fetched
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=False
        ):
            prices = self.broker.fetch_live_prices()

        self.assertAlmostEqual(prices["BTC"],  60_000.0)
        self.assertAlmostEqual(prices["ETH"],  3_000.0)
        self.assertAlmostEqual(prices["SOL"],  150.0)
        self.assertAlmostEqual(prices["XRP"],  1.50)

    def test_etf_prices_skipped_outside_market_hours(self):
        self.broker._api.get_latest_crypto_trades.return_value = {}
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=False
        ):
            self.broker.fetch_live_prices()
        self.broker._api.get_latest_trades.assert_not_called()

    def test_etf_prices_fetched_during_market_hours(self):
        etf_trades = {
            "QLD":  _mock_trade(80.0),
            "TQQQ": _mock_trade(50.0),
            "SSO":  _mock_trade(70.0),
            "SQQQ": _mock_trade(30.0),
            "SH":   _mock_trade(15.0),
        }
        self.broker._api.get_latest_crypto_trades.return_value = {}
        self.broker._api.get_latest_trades.return_value = etf_trades
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=True
        ):
            prices = self.broker.fetch_live_prices()
        self.assertIn("ETHD", prices)   # internal name
        self.assertAlmostEqual(prices["ETHD"], 30.0)   # SQQQ price
        self.assertIn("SETH", prices)
        self.assertAlmostEqual(prices["SETH"], 15.0)   # SH price

    def test_prices_are_fresh_after_fetch(self):
        self.broker._api.get_latest_crypto_trades.return_value = {
            "BTC/USD": _mock_trade(55_000.0),
        }
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=False
        ):
            self.broker.fetch_live_prices()
        self.assertTrue(self.broker.prices_are_fresh(max_age_sec=5.0))

    def test_prices_stale_after_timeout(self):
        self.broker.live_prices = {"BTC": 50_000.0}
        self.broker.last_price_timestamp = time.time() - 20
        self.assertFalse(self.broker.prices_are_fresh(max_age_sec=10.0))

    def test_prices_not_fresh_when_empty(self):
        self.broker.live_prices = {}
        self.assertFalse(self.broker.prices_are_fresh())


# ---------------------------------------------------------------------------
# Tests: get_positions / sync
# ---------------------------------------------------------------------------

class TestGetPositions(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker()

    def test_returns_position_dict(self):
        pos_btc = MagicMock(); pos_btc.symbol = "BTC/USD"; pos_btc.qty = "0.5"
        pos_sqqq = MagicMock(); pos_sqqq.symbol = "SQQQ";  pos_sqqq.qty = "10"
        self.broker._api.list_positions.return_value = [pos_btc, pos_sqqq]
        positions = self.broker.get_positions()
        self.assertAlmostEqual(positions["BTC/USD"], 0.5)
        self.assertAlmostEqual(positions["SQQQ"],    10.0)

    def test_syncs_etf_positions_dict(self):
        pos_sqqq = MagicMock(); pos_sqqq.symbol = "SQQQ"; pos_sqqq.qty = "7.5"
        self.broker._api.list_positions.return_value = [pos_sqqq]
        self.broker.get_positions()
        # SQQQ is the Alpaca symbol for internal "ETHD"
        self.assertAlmostEqual(self.broker.etf_positions["ETHD"], 7.5)

    def test_syncs_crypto_positions_dict(self):
        pos_btc = MagicMock(); pos_btc.symbol = "BTC/USD"; pos_btc.qty = "0.25"
        self.broker._api.list_positions.return_value = [pos_btc]
        self.broker.get_positions()
        self.assertAlmostEqual(self.broker.spot_positions["BTC"], 0.25)

    def test_returns_cached_on_api_error(self):
        self.broker.live_positions = {"BTC/USD": 1.0}
        self.broker._api.list_positions.side_effect = Exception("timeout")
        positions = self.broker.get_positions()
        self.assertAlmostEqual(positions["BTC/USD"], 1.0)


# ---------------------------------------------------------------------------
# Tests: ETF overlay
# ---------------------------------------------------------------------------

class TestEtfOverlay(unittest.TestCase):

    def setUp(self):
        self.broker = _make_broker(dry_run=True)
        # Prime equity + ETF prices
        self.broker.live_balances = {"equity": 10_000.0}
        self.broker._api.get_account.return_value = _mock_account(equity=10_000.0)
        # Set ETF prices via live_prices so ETFHedger can compute targets
        self.broker.live_prices = {
            "ETHD": 30.0,
            "SETH": 15.0,
            "ETHU": 80.0,
            "SLON": 50.0,
            "XXRP": 70.0,
        }
        self.broker.etf_positions = {
            "ETHU": 0.0, "SLON": 0.0, "XXRP": 0.0,
            "ETHD": 0.0, "SETH": 0.0,
        }

    def test_overlay_runs_during_market_hours(self):
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=True
        ), patch.object(
            self.broker._market_hours, "required_order_type", return_value="market"
        ):
            # Regime: severe panic → ETFHedger should want SETH position
            regime = {"panic_risk": 2, "cycle_phase": 3}
            self.broker.run_etf_overlay(agent=None, prices={}, regime=regime)
        # In dry_run mode no real order goes out, but etf_positions may be updated
        # (AlpacaBroker updates positions on successful place_order dry-run result)
        # Just verify the method ran without exception and kill switch is off
        self.assertFalse(self.broker.kill_switch)

    def test_overlay_skipped_outside_market_hours(self):
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=False
        ):
            self.broker.run_etf_overlay(agent=None, prices={}, regime={})
        # No positions changed
        self.assertEqual(self.broker.etf_positions["ETHD"], 0.0)

    def test_overlay_skipped_when_kill_switch_active(self):
        self.broker.kill_switch = True
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=True
        ):
            self.broker.run_etf_overlay(agent=None, prices={}, regime={})
        # Verify no API call was attempted
        self.broker._api.submit_order.assert_not_called()

    def test_overlay_uses_alpaca_etf_symbols(self):
        """
        In non-dry-run mode, orders for ETHD must use the Alpaca symbol SQQQ
        and orders for SETH must use SH.
        """
        self.broker.dry_run = False
        self.broker.live_balances = {
            "equity": 10_000.0, "pattern_day_trader": False,
            "daytrade_count": 0, "account_type": "MARGIN",
        }
        self.broker._api.get_account.return_value = _mock_account(equity=10_000.0)
        self.broker._api.submit_order.return_value = _mock_order()
        # Force market hours open and required_order_type = "market"
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=True
        ), patch.object(
            self.broker._market_hours, "required_order_type", return_value="market"
        ):
            regime = {"panic_risk": 2, "cycle_phase": 3}  # strong bear signal
            self.broker.run_etf_overlay(agent=None, prices={}, regime=regime)

        submitted_symbols = [
            call[1].get("symbol") or call[0][0]
            for call in self.broker._api.submit_order.call_args_list
        ]
        # Should have submitted orders using Alpaca symbols (SQQQ, SH, not ETHD/SETH)
        for sym in submitted_symbols:
            self.assertNotIn(sym, {"ETHD", "SETH"},
                             f"Order submitted with internal ticker {sym!r} instead of Alpaca symbol")
            self.assertIn(sym, set(ALPACA_ETF_PAIRS.values()),
                          f"Unexpected Alpaca symbol {sym!r}")

    def test_neutral_regime_holds_existing_positions(self):
        """No new sell orders should fire on a neutral signal."""
        # Give broker an existing SETH position
        self.broker.etf_positions["SETH"] = 5.0
        with patch.object(
            self.broker._market_hours, "etf_trading_allowed", return_value=True
        ), patch.object(
            self.broker._market_hours, "required_order_type", return_value="market"
        ):
            regime = {"macro_regime": 0.0, "bullish_confidence": 0.3,
                      "panic_risk": 0, "bearish_drift": False, "cycle_phase": 1}
            self.broker.run_etf_overlay(agent=None, prices={}, regime=regime)
        # Since the regime is neutral, any sell order should be held (not placed)
        sell_calls = [
            c for c in self.broker._api.submit_order.call_args_list
            if c[1].get("side") == "sell" or (c[0] and c[0][1] == "sell")
        ]
        self.assertEqual(len(sell_calls), 0,
                         "Sell orders were placed despite neutral signal")


# ---------------------------------------------------------------------------
# Tests: validate_credentials
# ---------------------------------------------------------------------------

class TestValidateCredentials(unittest.TestCase):

    def test_returns_true_on_success(self):
        broker = _make_broker()
        broker._api.get_account.return_value = _mock_account(equity=25_000.0)
        self.assertTrue(broker.validate_credentials())

    def test_returns_false_on_api_error(self):
        broker = _make_broker()
        broker._api.get_account.side_effect = Exception("invalid key")
        self.assertFalse(broker.validate_credentials())

    def test_returns_false_with_no_client(self):
        broker = _make_broker()
        broker._api = None
        self.assertFalse(broker.validate_credentials())


# ---------------------------------------------------------------------------
# Tests: create_alpaca_broker factory
# ---------------------------------------------------------------------------

class TestFactory(unittest.TestCase):

    def test_factory_dry_run_by_default(self):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY":       "k",
            "ALPACA_API_SECRET":    "s",
            "ENABLE_LIVE_TRADING":  "false",
        }):
            broker = create_alpaca_broker()
        self.assertTrue(broker.dry_run)

    def test_factory_live_when_env_set(self):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY":       "k",
            "ALPACA_API_SECRET":    "s",
            "ENABLE_LIVE_TRADING":  "true",
        }):
            # Live trading requires BOTH dry_run=False AND ENABLE_LIVE_TRADING=true
            broker = create_alpaca_broker(dry_run=False)
        self.assertFalse(broker.dry_run)


if __name__ == "__main__":
    unittest.main()
