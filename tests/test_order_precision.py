"""
Unit tests for Kraken order price/volume precision formatting.

These tests verify that _format_order_price and _format_order_volume
produce strings that satisfy Kraken's per-pair decimal-place requirements,
ensuring that orders are never rejected for invalid precision
(e.g. EOrder:Invalid price:SOL/USD price can only be specified up to 2 decimals).
"""
import sys
import os
import unittest

# Allow importing from project/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from broker.broker import (
    _format_order_price,
    _format_order_volume,
    KRAKEN_PRICE_DECIMALS,
    KRAKEN_VOLUME_DECIMALS,
    _FALLBACK_PRICE_DECIMALS,
    _FALLBACK_VOLUME_DECIMALS,
)


class TestFormatOrderPrice(unittest.TestCase):
    """Tests for _format_order_price()."""

    # ------------------------------------------------------------------
    # SOL/USD — the pair that triggered the kill switch in production
    # ------------------------------------------------------------------

    def test_sol_usd_exact_two_dp(self):
        """SOL/USD requires exactly 2 decimal places."""
        result = _format_order_price("SOLUSD", 91.14)
        self.assertEqual(result, "91.14")
        self.assertEqual(len(result.split(".")[-1]), 2)

    def test_sol_usd_rounds_excess_decimals(self):
        """A price with >2 dp is rounded down to 2 dp (the bug case)."""
        # This was the price that caused the Kraken rejection: 91.1400001
        result = _format_order_price("SOLUSD", 91.1400001)
        self.assertEqual(result, "91.14")

    def test_sol_usd_rounds_up_correctly(self):
        result = _format_order_price("SOLUSD", 91.145)
        # round half-even: 91.14 or 91.15 depending on Python rounding, but always 2 dp
        self.assertRegex(result, r"^\d+\.\d{2}$")

    def test_sol_usd_zero_cents(self):
        result = _format_order_price("SOLUSD", 100.0)
        self.assertEqual(result, "100.00")

    # ------------------------------------------------------------------
    # BTC/USD — 1 decimal place
    # ------------------------------------------------------------------

    def test_btc_usd_one_dp(self):
        result = _format_order_price("XXBTZUSD", 70713.1)
        self.assertEqual(result, "70713.1")
        self.assertEqual(len(result.split(".")[-1]), 1)

    def test_btc_usd_rounds_excess(self):
        result = _format_order_price("XXBTZUSD", 70713.15)
        self.assertRegex(result, r"^\d+\.\d{1}$")

    # ------------------------------------------------------------------
    # ETH/USD — 2 decimal places
    # ------------------------------------------------------------------

    def test_eth_usd_two_dp(self):
        result = _format_order_price("XETHZUSD", 2160.06)
        self.assertEqual(result, "2160.06")

    # ------------------------------------------------------------------
    # AVAX/USD — 2 decimal places
    # ------------------------------------------------------------------

    def test_avax_usd_two_dp(self):
        result = _format_order_price("AVAXUSD", 9.62)
        self.assertEqual(result, "9.62")

    # ------------------------------------------------------------------
    # LINK/USD — 4 decimal places
    # ------------------------------------------------------------------

    def test_link_usd_four_dp(self):
        result = _format_order_price("LINKUSD", 9.2500)
        self.assertEqual(result, "9.2500")

    # ------------------------------------------------------------------
    # HBAR/USD — 5 decimal places
    # ------------------------------------------------------------------

    def test_hbar_usd_five_dp(self):
        result = _format_order_price("HBARUSD", 0.09123)
        self.assertEqual(result, "0.09123")

    # ------------------------------------------------------------------
    # XRP/USD — 5 decimal places
    # ------------------------------------------------------------------

    def test_xrp_usd_five_dp(self):
        result = _format_order_price("XXRPZUSD", 1.41730)
        self.assertEqual(result, "1.41730")

    def test_xrp_usd_rounds_excess(self):
        result = _format_order_price("XXRPZUSD", 1.4173012345)
        self.assertRegex(result, r"^\d+\.\d{5}$")

    # ------------------------------------------------------------------
    # XLM/USD — 6 decimal places
    # ------------------------------------------------------------------

    def test_xlm_usd_six_dp(self):
        result = _format_order_price("XXLMZUSD", 0.178100)
        self.assertEqual(result, "0.178100")

    # ------------------------------------------------------------------
    # ETF/ETP tokens — 2 decimal places
    # ------------------------------------------------------------------

    def test_ethu_usd_two_dp(self):
        result = _format_order_price("ETHUUSD", 2160.06)
        self.assertEqual(result, "2160.06")

    def test_seth_usd_two_dp(self):
        result = _format_order_price("SETHUSD", 2160.06)
        self.assertEqual(result, "2160.06")

    def test_ethd_usd_two_dp(self):
        result = _format_order_price("ETHDUSD", 2160.06)
        self.assertEqual(result, "2160.06")

    # ------------------------------------------------------------------
    # Fallback for unknown pairs
    # ------------------------------------------------------------------

    def test_unknown_pair_falls_back_to_two_dp(self):
        """Unknown pairs use the safe fallback precision (2 dp)."""
        result = _format_order_price("UNKNOWNUSD", 10.12345)
        self.assertEqual(result, "10.12")
        self.assertEqual(_FALLBACK_PRICE_DECIMALS, 2)

    # ------------------------------------------------------------------
    # Result is always a string (Kraken API expects string fields)
    # ------------------------------------------------------------------

    def test_returns_string(self):
        self.assertIsInstance(_format_order_price("SOLUSD", 91.14), str)


class TestFormatOrderVolume(unittest.TestCase):
    """Tests for _format_order_volume()."""

    def test_sol_volume_eight_dp(self):
        result = _format_order_volume("SOLUSD", 0.11443653)
        self.assertEqual(result, "0.11443653")
        self.assertEqual(len(result.split(".")[-1]), 8)

    def test_btc_volume_eight_dp(self):
        result = _format_order_volume("XXBTZUSD", 0.00019234)
        self.assertEqual(result, "0.00019234")

    def test_xrp_volume_eight_dp(self):
        result = _format_order_volume("XXRPZUSD", 47.26996223)
        self.assertEqual(result, "47.26996223")

    def test_unknown_pair_falls_back_to_eight_dp(self):
        result = _format_order_volume("UNKNOWNUSD", 1.12345678)
        self.assertEqual(result, "1.12345678")
        self.assertEqual(_FALLBACK_VOLUME_DECIMALS, 8)

    def test_returns_string(self):
        self.assertIsInstance(_format_order_volume("SOLUSD", 0.5), str)


class TestPrecisionDicts(unittest.TestCase):
    """Sanity checks on the precision dictionaries themselves."""

    def test_all_spot_pairs_have_price_entry(self):
        """All Kraken spot pair symbols used by the bot have price entries."""
        required = {"XXBTZUSD", "XETHZUSD", "SOLUSD", "AVAXUSD",
                    "LINKUSD", "HBARUSD", "XXRPZUSD", "XXLMZUSD"}
        for pair in required:
            self.assertIn(pair, KRAKEN_PRICE_DECIMALS,
                          f"Missing price precision entry for {pair}")

    def test_all_spot_pairs_have_volume_entry(self):
        required = {"XXBTZUSD", "XETHZUSD", "SOLUSD", "AVAXUSD",
                    "LINKUSD", "HBARUSD", "XXRPZUSD", "XXLMZUSD"}
        for pair in required:
            self.assertIn(pair, KRAKEN_VOLUME_DECIMALS,
                          f"Missing volume precision entry for {pair}")

    def test_sol_price_precision_is_two(self):
        """SOL/USD must be exactly 2 dp — this is what Kraken enforces."""
        self.assertEqual(KRAKEN_PRICE_DECIMALS["SOLUSD"], 2)

    def test_btc_price_precision_is_one(self):
        self.assertEqual(KRAKEN_PRICE_DECIMALS["XXBTZUSD"], 1)

    def test_precision_values_are_positive_ints(self):
        for pair, dp in KRAKEN_PRICE_DECIMALS.items():
            self.assertIsInstance(dp, int, f"{pair}: dp must be int")
            self.assertGreater(dp, 0, f"{pair}: dp must be > 0")
        for pair, dp in KRAKEN_VOLUME_DECIMALS.items():
            self.assertIsInstance(dp, int, f"{pair}: dp must be int")
            self.assertGreater(dp, 0, f"{pair}: dp must be > 0")


if __name__ == "__main__":
    unittest.main()
