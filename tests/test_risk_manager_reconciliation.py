"""
Tests for RiskManager reconciliation with new ETF/allocation/exit logic.

Verifies that:
1. ETF skips (neutral regime, min-notional, timeout recovery) are tracked
   separately from risk violations.
2. Strategy holds (fee-hurdle, neutral regime) are tracked and reported
   distinctly from risk blocks in episode_summary().
3. register_agents() correctly resets all ETF/strategy event counters.
4. episode_summary() includes both risk-violation keys and strategy-event keys.
5. LiveBroker notional caps scale with equity via _sync_safety_caps_to_equity().
"""
import sys
import os
import unittest
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from risk.risk_manager import RiskManager, OrderIntent


# ---------------------------------------------------------------------------
# Minimal agent stub — only the fields RiskManager actually reads.
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, name: str, balance: float = 1_000.0, position: float = 0.0):
        self.name = name
        self.balance = balance
        self.position = position
        self.unrealized_pnl = 0.0


class _FakeMarketState:
    def __init__(self, mid_price: float = 100.0):
        self.mid_price = mid_price


# ---------------------------------------------------------------------------
# Tests for ETF skip tracking
# ---------------------------------------------------------------------------

class TestETFSkipTracking(unittest.TestCase):
    """RiskManager.record_etf_skip() stores events under a separate key."""

    def setUp(self):
        self.rm = RiskManager()
        self.agent = _FakeAgent("RL", balance=5_000.0)
        self.rm.register_agents([self.agent])

    def test_record_etf_skip_stored(self):
        self.rm.record_etf_skip("RL", "neutral_regime", {"net_exposure": 0.001})
        self.assertEqual(len(self.rm.etf_skips["RL"]), 1)
        self.assertEqual(self.rm.etf_skips["RL"][0]["reason"], "neutral_regime")

    def test_record_multiple_etf_skips(self):
        for reason in ("neutral_regime", "min_notional", "timeout_recovery", "cap_breached"):
            self.rm.record_etf_skip("RL", reason)
        self.assertEqual(len(self.rm.etf_skips["RL"]), 4)

    def test_etf_skip_not_in_rejected_orders(self):
        """ETF skips must NOT count as rejected/blocked orders."""
        self.rm.record_etf_skip("RL", "neutral_regime")
        self.assertEqual(len(self.rm.rejected_orders.get("RL", [])), 0)

    def test_unknown_agent_auto_created(self):
        """record_etf_skip should handle agent names not in register_agents."""
        self.rm.record_etf_skip("ETF_LAYER", "market_closed")
        self.assertEqual(len(self.rm.etf_skips["ETF_LAYER"]), 1)

    def test_episode_summary_includes_etf_skips(self):
        self.rm.record_etf_skip("RL", "neutral_regime")
        summary = self.rm.episode_summary()
        self.assertIn("etf_skips_per_agent", summary)
        self.assertEqual(summary["etf_skips_per_agent"].get("RL", 0), 1)

    def test_episode_summary_etf_skips_empty_when_none(self):
        """If no ETF skips occurred, etf_skips_per_agent should be an empty dict."""
        summary = self.rm.episode_summary()
        self.assertIn("etf_skips_per_agent", summary)
        self.assertEqual(summary["etf_skips_per_agent"], {})


# ---------------------------------------------------------------------------
# Tests for strategy-hold tracking
# ---------------------------------------------------------------------------

class TestStrategyHoldTracking(unittest.TestCase):
    """RiskManager.record_strategy_hold() stores holds under a separate key."""

    def setUp(self):
        self.rm = RiskManager()
        self.agent = _FakeAgent("RL", balance=5_000.0)
        self.rm.register_agents([self.agent])

    def test_record_strategy_hold_stored(self):
        self.rm.record_strategy_hold("RL", "fee_hurdle", {"notional": 15.0, "hurdle": 40.0})
        self.assertEqual(len(self.rm.strategy_holds["RL"]), 1)
        self.assertEqual(self.rm.strategy_holds["RL"][0]["reason"], "fee_hurdle")

    def test_record_neutral_regime_hold(self):
        self.rm.record_strategy_hold("RL", "neutral_regime")
        self.assertEqual(len(self.rm.strategy_holds["RL"]), 1)

    def test_strategy_hold_not_in_rejected_orders(self):
        """Strategy holds must NOT count as rejected/blocked orders."""
        self.rm.record_strategy_hold("RL", "fee_hurdle")
        self.assertEqual(len(self.rm.rejected_orders.get("RL", [])), 0)

    def test_episode_summary_includes_strategy_holds(self):
        self.rm.record_strategy_hold("RL", "fee_hurdle")
        self.rm.record_strategy_hold("RL", "neutral_regime")
        summary = self.rm.episode_summary()
        self.assertIn("strategy_holds_per_agent", summary)
        self.assertEqual(summary["strategy_holds_per_agent"].get("RL", 0), 2)

    def test_episode_summary_holds_empty_when_none(self):
        summary = self.rm.episode_summary()
        self.assertEqual(summary["strategy_holds_per_agent"], {})


# ---------------------------------------------------------------------------
# Tests for register_agents() resetting ETF/strategy counters
# ---------------------------------------------------------------------------

class TestRegisterAgentsResetsETFCounters(unittest.TestCase):
    """register_agents() must clear ETF skips and strategy holds each episode."""

    def test_etf_skips_cleared_on_new_episode(self):
        rm = RiskManager()
        agent = _FakeAgent("RL")
        rm.register_agents([agent])
        rm.record_etf_skip("RL", "neutral_regime")
        self.assertEqual(len(rm.etf_skips["RL"]), 1)

        # New episode
        rm.register_agents([agent])
        self.assertEqual(rm.etf_skips.get("RL", []), [])

    def test_strategy_holds_cleared_on_new_episode(self):
        rm = RiskManager()
        agent = _FakeAgent("RL")
        rm.register_agents([agent])
        rm.record_strategy_hold("RL", "fee_hurdle")
        rm.register_agents([agent])
        self.assertEqual(rm.strategy_holds.get("RL", []), [])

    def test_rejected_orders_also_cleared(self):
        rm = RiskManager()
        agent = _FakeAgent("RL", balance=100.0)
        rm.register_agents([agent])
        # Trigger a notional rejection manually
        intent = OrderIntent(side="buy", quantity=1_000_000, price=100.0)
        state = _FakeMarketState(mid_price=100.0)
        rm.approve_order(agent, intent, state)
        self.assertGreater(len(rm.rejected_orders.get("RL", [])), 0)

        rm.register_agents([agent])
        self.assertEqual(rm.rejected_orders.get("RL", []), [])


# ---------------------------------------------------------------------------
# Tests for episode_summary() completeness
# ---------------------------------------------------------------------------

class TestEpisodeSummarySchema(unittest.TestCase):
    """episode_summary() must include all expected keys."""

    REQUIRED_KEYS = {
        "rejected_per_agent",
        "drawdown_locked",
        "global_kill_switch",
        "etf_skips_per_agent",
        "strategy_holds_per_agent",
    }

    def test_all_keys_present_empty_episode(self):
        rm = RiskManager()
        rm.register_agents([_FakeAgent("A")])
        summary = rm.episode_summary()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, summary, f"Missing key: {key}")

    def test_all_keys_present_with_events(self):
        rm = RiskManager()
        agent = _FakeAgent("A", balance=500.0)
        rm.register_agents([agent])
        rm.record_etf_skip("A", "neutral_regime")
        rm.record_strategy_hold("A", "fee_hurdle")
        intent = OrderIntent(side="buy", quantity=1_000_000, price=100.0)
        rm.approve_order(agent, intent, _FakeMarketState(100.0))

        summary = rm.episode_summary()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, summary, f"Missing key: {key}")
        self.assertGreater(summary["rejected_per_agent"].get("A", 0), 0)
        self.assertEqual(summary["etf_skips_per_agent"].get("A", 0), 1)
        self.assertEqual(summary["strategy_holds_per_agent"].get("A", 0), 1)

    def test_risk_blocks_do_not_appear_in_strategy_keys(self):
        """A true risk block must appear in rejected_per_agent, NOT in etf_skips."""
        rm = RiskManager()
        agent = _FakeAgent("B", balance=500.0)
        rm.register_agents([agent])
        # This will be rejected by the notional limit
        intent = OrderIntent(side="buy", quantity=1_000_000, price=100.0)
        rm.approve_order(agent, intent, _FakeMarketState(100.0))
        summary = rm.episode_summary()
        self.assertGreater(summary["rejected_per_agent"].get("B", 0), 0)
        self.assertEqual(summary["etf_skips_per_agent"].get("B", 0), 0)
        self.assertEqual(summary["strategy_holds_per_agent"].get("B", 0), 0)


# ---------------------------------------------------------------------------
# Tests for LiveBroker notional cap scaling
# ---------------------------------------------------------------------------

class TestLiveBrokerNotionalCapScaling(unittest.TestCase):
    """
    LiveBroker._sync_safety_caps_to_equity() must scale the notional caps
    proportionally to account equity so ETF orders on larger accounts are
    not blocked by the conservative first-night defaults ($50 per asset).
    """

    def _make_minimal_live_broker(self, zusd: float = 500.0):
        """
        Build a minimal LiveBroker stub with just the fields needed to test
        _sync_safety_caps_to_equity().  We monkey-patch compute_total_equity()
        to return a fixed equity so no Kraken API calls are made.
        """
        import broker.broker as broker_mod
        LiveBroker = broker_mod.LiveBroker

        # Create instance without calling __init__ (avoids network/env deps)
        obj = object.__new__(LiveBroker)

        # Seed only the attributes _sync_safety_caps_to_equity() needs
        obj.max_notional_per_asset  = 50.0
        obj.max_total_notional      = 200.0
        obj._max_notional_pct       = 0.35
        obj._max_total_notional_pct = 2.0
        obj._per_asset_cap_fixed    = False
        obj._total_cap_fixed        = False
        obj._equity_for_test        = zusd

        # Patch compute_total_equity to return our test equity
        obj.compute_total_equity = lambda: obj._equity_for_test

        return obj

    def test_caps_scale_to_equity(self):
        broker = self._make_minimal_live_broker(zusd=500.0)
        broker._sync_safety_caps_to_equity()
        # 35% of $500 = $175; 200% of $500 = $1000
        self.assertAlmostEqual(broker.max_notional_per_asset, 500.0 * 0.35, places=4)
        self.assertAlmostEqual(broker.max_total_notional,     500.0 * 2.0,  places=4)

    def test_etf_allocation_fits_within_scaled_cap(self):
        """
        For a $500 account: 30% ETF allocation = $150 must not exceed the scaled cap.
        """
        broker = self._make_minimal_live_broker(zusd=500.0)
        broker._sync_safety_caps_to_equity()
        etf_allocation = 500.0 * 0.30   # $150
        self.assertLessEqual(etf_allocation, broker.max_notional_per_asset,
                             "ETF allocation should fit within scaled per-asset cap")

    def test_caps_not_scaled_when_fixed(self):
        """
        When _per_asset_cap_fixed=True (hard env-var override), only the
        per-asset cap stays unchanged.  The total cap still scales.
        """
        broker = self._make_minimal_live_broker(zusd=500.0)
        broker._per_asset_cap_fixed = True
        broker._sync_safety_caps_to_equity()
        # per-asset cap must remain at its original $50 (env-var locked)
        self.assertAlmostEqual(broker.max_notional_per_asset, 50.0)
        # total cap should still be scaled
        self.assertAlmostEqual(broker.max_total_notional, 500.0 * 2.0, places=4)

    def test_total_cap_not_scaled_when_fixed(self):
        """
        When _total_cap_fixed=True, only the total cap stays unchanged.
        """
        broker = self._make_minimal_live_broker(zusd=500.0)
        broker._total_cap_fixed = True
        broker._sync_safety_caps_to_equity()
        self.assertAlmostEqual(broker.max_notional_per_asset, 500.0 * 0.35, places=4)
        self.assertAlmostEqual(broker.max_total_notional, 200.0)

    def test_caps_not_scaled_when_equity_zero(self):
        """If equity is 0 (no balance yet), caps must remain unchanged."""
        broker = self._make_minimal_live_broker(zusd=0.0)
        broker._sync_safety_caps_to_equity()
        self.assertAlmostEqual(broker.max_notional_per_asset, 50.0)
        self.assertAlmostEqual(broker.max_total_notional, 200.0)

    def test_large_account_not_blocked(self):
        """
        On a $10,000 account the first-night $50 cap would block a $3,000 ETF
        order (30% of $10k).  After scaling it must be allowed.
        """
        broker = self._make_minimal_live_broker(zusd=10_000.0)
        broker._sync_safety_caps_to_equity()
        etf_allocation = 10_000.0 * 0.30   # $3,000
        self.assertLessEqual(etf_allocation, broker.max_notional_per_asset,
                             "30% ETF allocation must fit within equity-scaled cap")


if __name__ == "__main__":
    unittest.main()
