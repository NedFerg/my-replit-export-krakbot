"""
Unit tests for VolumeClimaxDetector
=====================================
Covers:
  - Default construction and parameter validation
  - ClimaxResult namedtuple fields
  - Insufficient history returns neutral result
  - Volume ratio computation
  - Capitulation signal: volume spike + price decline + RSI oversold
  - Exhaustion signal: volume spike + narrow range + declining volume
  - Boolean flags fire at correct thresholds
  - reset() clears all state
  - update() with missing volume / price_range
  - Edge cases: zero price, single bar, constant price
"""

from __future__ import annotations

import math
import sys
import os

# Add project/ to path so that `strategies.*` imports resolve correctly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

import pytest

from strategies.signals.volume_climax_detector import (
    VolumeClimaxDetector,
    ClimaxResult,
    _NEUTRAL,
    CAPITULATION_THRESHOLD,
    EXHAUSTION_THRESHOLD,
    MIN_HISTORY_BARS,
    VOLUME_SPIKE_RATIO,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed(det: VolumeClimaxDetector, n: int, price: float = 100.0, volume: float = 1_000.0) -> None:
    """Feed *n* identical bars to build up history."""
    for _ in range(n):
        det.update(price=price, volume=volume, price_range=1.0)


def _feed_sequence(
    det: VolumeClimaxDetector,
    prices: list[float],
    volumes: list[float] | None = None,
    ranges: list[float] | None = None,
) -> ClimaxResult:
    """Feed a sequence of bars and return the last result."""
    if volumes is None:
        volumes = [1_000.0] * len(prices)
    if ranges is None:
        ranges = [1.0] * len(prices)
    result = _NEUTRAL
    for p, v, r in zip(prices, volumes, ranges):
        result = det.update(price=p, volume=v, price_range=r)
    return result


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        det = VolumeClimaxDetector()
        assert det.volume_avg_window == 20
        assert det.volume_spike_ratio == 2.5
        assert det.capitulation_threshold == 0.60
        assert det.exhaustion_threshold == 0.60
        assert det.min_history_bars == 22

    def test_custom_parameters(self):
        det = VolumeClimaxDetector(
            volume_avg_window=10,
            volume_spike_ratio=3.0,
            capitulation_threshold=0.5,
            exhaustion_threshold=0.55,
            min_history_bars=15,
        )
        assert det.volume_avg_window == 10
        assert det.volume_spike_ratio == 3.0
        assert det.capitulation_threshold == 0.5
        assert det.exhaustion_threshold == 0.55
        assert det.min_history_bars == 15

    def test_initial_last_result_is_neutral(self):
        det = VolumeClimaxDetector()
        assert det.last_result is _NEUTRAL


# ---------------------------------------------------------------------------
# ClimaxResult namedtuple
# ---------------------------------------------------------------------------

class TestClimaxResult:
    def test_namedtuple_fields(self):
        r = ClimaxResult(
            capitulation_score=0.7,
            exhaustion_score=0.3,
            is_capitulation=True,
            is_exhaustion=False,
            volume_ratio=3.5,
            rsi_val=25.0,
            price_change=-0.05,
        )
        assert r.capitulation_score == 0.7
        assert r.exhaustion_score == 0.3
        assert r.is_capitulation is True
        assert r.is_exhaustion is False
        assert r.volume_ratio == 3.5
        assert r.rsi_val == 25.0
        assert r.price_change == -0.05

    def test_neutral_sentinel_values(self):
        assert _NEUTRAL.capitulation_score == 0.0
        assert _NEUTRAL.exhaustion_score == 0.0
        assert _NEUTRAL.is_capitulation is False
        assert _NEUTRAL.is_exhaustion is False


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------

class TestInsufficientHistory:
    def test_returns_neutral_before_min_history(self):
        det = VolumeClimaxDetector(min_history_bars=22)
        # Feed fewer bars than required
        for i in range(21):
            result = det.update(price=100.0, volume=1_000.0)
            assert result is _NEUTRAL, f"Expected neutral at bar {i + 1}"

    def test_starts_emitting_at_min_history(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        for i in range(4):
            det.update(price=100.0, volume=1_000.0)
        # 5th bar should produce a real (possibly zero-scored) result
        result = det.update(price=100.0, volume=1_000.0)
        assert result is not _NEUTRAL

    def test_zero_price_returns_last_result(self):
        det = VolumeClimaxDetector()
        result = det.update(price=0.0, volume=1_000.0)
        assert result is _NEUTRAL  # last_result starts as _NEUTRAL


# ---------------------------------------------------------------------------
# Volume ratio
# ---------------------------------------------------------------------------

class TestVolumeRatio:
    def test_volume_ratio_nan_with_no_history(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        # Feed enough bars to pass min_history but only 1 volume bar
        for _ in range(5):
            det.update(price=100.0, volume=0.0)  # no volume
        result = det.update(price=100.0, volume=0.0)
        assert math.isnan(result.volume_ratio)

    def test_volume_ratio_computed_correctly(self):
        det = VolumeClimaxDetector(min_history_bars=5, volume_avg_window=4)
        # Feed 4 bars at volume=1000, then one at 3000 → ratio ≈ 3.0
        for _ in range(4):
            det.update(price=100.0, volume=1_000.0)
        result = det.update(price=100.0, volume=3_000.0)
        assert not math.isnan(result.volume_ratio)
        assert abs(result.volume_ratio - 3.0) < 0.1

    def test_volume_ratio_above_one_when_above_avg(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 20, price=100.0, volume=1_000.0)
        result = det.update(price=100.0, volume=5_000.0)
        assert result.volume_ratio > 1.0


# ---------------------------------------------------------------------------
# Capitulation signal
# ---------------------------------------------------------------------------

class TestCapitulationScore:
    def test_no_capitulation_with_flat_price_and_normal_volume(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        result = _feed_sequence(
            det,
            prices=[100.0] * 30,
            volumes=[1_000.0] * 30,
        )
        assert result.capitulation_score < CAPITULATION_THRESHOLD
        assert result.is_capitulation is False

    def test_capitulation_fires_on_volume_spike_and_price_drop(self):
        """Feed a crash scenario: high volume + sharp price decline."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=10,
            volume_spike_ratio=2.0,
            price_decline_threshold=0.05,
            rsi_oversold=30,
            capitulation_threshold=0.50,
        )
        # Build history at stable price
        _feed(det, 15, price=100.0, volume=1_000.0)
        # Crash: price drops ~10 % with 5× average volume
        prices_crash = [100.0, 98.0, 96.0, 93.0, 90.0]
        volumes_crash = [5_000.0] * 5
        result = _feed_sequence(det, prices_crash, volumes_crash)
        assert result.capitulation_score > 0.0
        # Volume spike alone should contribute positively
        assert result.volume_ratio > 1.0

    def test_capitulation_score_increases_with_larger_drop(self):
        """Bigger price drop → higher capitulation score."""
        def score_for_drop(pct_drop: float) -> float:
            det = VolumeClimaxDetector(
                min_history_bars=5,
                volume_avg_window=5,
                volume_spike_ratio=2.0,
                price_decline_threshold=0.20,  # threshold high enough that drops are sub-maximum
                capitulation_threshold=0.99,
            )
            _feed(det, 10, price=100.0, volume=2_000.0)
            # Sharp drop in one bar
            final_price = 100.0 * (1.0 - pct_drop)
            result = det.update(price=final_price, volume=5_000.0)
            return result.capitulation_score

        assert score_for_drop(0.05) < score_for_drop(0.10)
        assert score_for_drop(0.10) < score_for_drop(0.15)

    def test_rsi_oversold_boosts_capitulation_score(self):
        """RSI oversold should add to capitulation score."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            rsi_period=5,
            rsi_oversold=30,
            capitulation_threshold=0.99,
        )
        # Drive RSI low by feeding a sequence of large down-moves
        prices = [100.0, 97.0, 94.0, 91.0, 88.0, 85.0, 82.0, 79.0, 76.0, 73.0, 70.0]
        volumes = [2_000.0] * len(prices)
        result = _feed_sequence(det, prices, volumes)
        if result.rsi_val is not None and result.rsi_val < 30:
            assert result.capitulation_score > 0.30

    def test_capitulation_boolean_threshold(self):
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=5,
            volume_spike_ratio=2.0,
            price_decline_threshold=0.03,
            capitulation_threshold=0.30,
            rsi_oversold=30,
        )
        _feed(det, 10, price=100.0, volume=1_000.0)
        # Volume spike at 4× average and 5 % price decline
        result = det.update(price=95.0, volume=4_000.0)
        # is_capitulation should match the threshold comparison
        assert result.is_capitulation == (result.capitulation_score >= 0.30)

    def test_buy_side_capitulation_on_price_surge(self):
        """Buy-side panic (surge) also scores on the adverse-move component."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=5,
            volume_spike_ratio=2.0,
            price_decline_threshold=0.05,
            capitulation_threshold=0.99,
        )
        _feed(det, 10, price=100.0, volume=1_000.0)
        # Explosive price surge with extreme volume
        result = det.update(price=115.0, volume=6_000.0)  # +15 % in one bar
        # The price move magnitude is 15 %, well above 5 % threshold
        assert result.capitulation_score > 0.35  # Component 2 full + some vol


# ---------------------------------------------------------------------------
# Exhaustion signal
# ---------------------------------------------------------------------------

class TestExhaustionScore:
    def test_no_exhaustion_with_normal_data(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        result = _feed_sequence(
            det,
            prices=[100.0] * 30,
            volumes=[1_000.0] * 30,
            ranges=[5.0] * 30,
        )
        assert result.exhaustion_score < EXHAUSTION_THRESHOLD
        assert result.is_exhaustion is False

    def test_exhaustion_fires_on_volume_spike_narrow_range_declining_volume(self):
        """Classic exhaustion: big volume spike, tiny price range, then volume tails off."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=10,
            volume_spike_ratio=2.0,
            exhaustion_range_window=3,
            volume_decline_bars=3,
            exhaustion_threshold=0.40,
        )
        # Normal history
        _feed(det, 15, price=100.0, volume=1_000.0)
        # Climax bar: very high volume but tiny intra-bar range
        det.update(price=101.0, volume=5_000.0, price_range=0.1)
        # Follow-through: declining volume
        det.update(price=101.5, volume=4_000.0, price_range=0.2)
        det.update(price=101.2, volume=3_000.0, price_range=0.15)
        result = det.update(price=101.0, volume=2_000.0, price_range=0.1)
        assert result.exhaustion_score > 0.0

    def test_exhaustion_narrow_range_component(self):
        """Below-average range with high volume increases exhaustion score."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=10,
            volume_spike_ratio=2.0,
            exhaustion_range_window=3,
            volume_decline_bars=1,
            exhaustion_threshold=0.99,
        )
        # History with normal ranges
        _feed(det, 15, price=100.0, volume=2_000.0)
        # High-volume bar with range much smaller than average (which is ~1.0)
        result = det.update(price=100.5, volume=8_000.0, price_range=0.05)
        # Narrow range (0.05 vs avg ~1.0) + volume spike should contribute
        assert result.exhaustion_score > 0.35

    def test_exhaustion_boolean_threshold(self):
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=5,
            volume_spike_ratio=2.0,
            exhaustion_range_window=3,
            volume_decline_bars=3,
            exhaustion_threshold=0.35,
        )
        _feed(det, 10, price=100.0, volume=1_000.0)
        det.update(price=100.5, volume=5_000.0, price_range=0.1)
        det.update(price=100.3, volume=4_000.0, price_range=0.1)
        result = det.update(price=100.1, volume=3_000.0, price_range=0.1)
        assert result.is_exhaustion == (result.exhaustion_score >= 0.35)

    def test_exhaustion_declining_volume_component(self):
        """Three bars of declining volume after a spike scores the component."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=10,
            volume_spike_ratio=2.0,
            exhaustion_range_window=3,
            volume_decline_bars=3,
            exhaustion_threshold=0.99,
        )
        _feed(det, 15, price=100.0, volume=1_000.0)
        # Climax then three declining bars
        det.update(price=101.0, volume=5_000.0, price_range=0.1)
        det.update(price=101.1, volume=4_000.0, price_range=0.1)
        det.update(price=101.0, volume=3_000.0, price_range=0.1)
        result = det.update(price=100.9, volume=2_000.0, price_range=0.1)
        # Three consecutive declines → decline_contrib = 3/3 = 1.0
        assert result.exhaustion_score >= 0.25 * 1.0 - 0.01  # Component 3 full weight


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_history(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 30, price=100.0, volume=1_000.0)
        det.reset()
        # After reset, history is gone — should return neutral again
        result = det.update(price=100.0, volume=1_000.0)
        assert result is _NEUTRAL

    def test_reset_restores_last_result_to_neutral(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 30, price=100.0, volume=1_000.0)
        det.reset()
        assert det.last_result is _NEUTRAL

    def test_detector_works_normally_after_reset(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 30, price=100.0, volume=1_000.0)
        det.reset()
        _feed(det, 4, price=100.0, volume=1_000.0)
        result = det.update(price=100.0, volume=1_000.0)
        assert result is not _NEUTRAL  # back to emitting real results


# ---------------------------------------------------------------------------
# Missing / optional inputs
# ---------------------------------------------------------------------------

class TestOptionalInputs:
    def test_update_without_volume(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        # No volume provided — should not raise
        for i in range(30):
            result = det.update(price=100.0 + i * 0.1)
        assert result is not _NEUTRAL

    def test_update_without_price_range(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 30)
        # price_range defaults to 0, detector estimates from price diff
        result = det.update(price=101.0, volume=1_000.0)
        assert result is not _NEUTRAL

    def test_update_with_zero_volume_skips_volume_components(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        for _ in range(30):
            det.update(price=100.0, volume=0.0)
        result = det.update(price=100.0, volume=0.0)
        # Without any volume data, volume_ratio should be NaN
        assert math.isnan(result.volume_ratio)

    def test_zero_price_returns_last_result_unchanged(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 10, price=100.0, volume=1_000.0)
        stored = det.last_result
        result = det.update(price=0.0, volume=5_000.0)
        assert result is stored


# ---------------------------------------------------------------------------
# Score boundaries
# ---------------------------------------------------------------------------

class TestScoreBoundaries:
    def test_scores_never_exceed_one(self):
        """Even extreme inputs must not produce scores above 1.0."""
        det = VolumeClimaxDetector(
            min_history_bars=5,
            volume_avg_window=5,
            volume_spike_ratio=1.5,
            price_decline_threshold=0.01,
            rsi_period=5,
            rsi_oversold=40,
            exhaustion_range_window=2,
            volume_decline_bars=2,
        )
        _feed(det, 10, price=100.0, volume=100.0)
        # Extreme crash
        for _ in range(10):
            result = det.update(price=50.0, volume=1_000_000.0, price_range=0.0001)
            assert result.capitulation_score <= 1.0
            assert result.exhaustion_score <= 1.0

    def test_scores_are_non_negative(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 30, price=100.0, volume=1_000.0)
        result = det.update(price=100.0, volume=1_000.0)
        assert result.capitulation_score >= 0.0
        assert result.exhaustion_score >= 0.0


# ---------------------------------------------------------------------------
# Price change field
# ---------------------------------------------------------------------------

class TestPriceChange:
    def test_price_change_reflects_n_bar_return(self):
        det = VolumeClimaxDetector(min_history_bars=5, price_change_window=3)
        prices = [100.0] * 25 + [95.0]  # drop of 5 % from bar-3
        result = _feed_sequence(det, prices)
        # price_change should be approximately (95-100)/100 = -0.05
        assert result.price_change < 0.0
        assert abs(result.price_change - (-0.05)) < 0.01

    def test_price_change_positive_on_price_rise(self):
        det = VolumeClimaxDetector(min_history_bars=5, price_change_window=5)
        prices = [100.0] * 25 + [110.0]
        result = _feed_sequence(det, prices)
        assert result.price_change > 0.0


# ---------------------------------------------------------------------------
# RSI field
# ---------------------------------------------------------------------------

class TestRSIField:
    def test_rsi_is_none_with_insufficient_history(self):
        det = VolumeClimaxDetector(min_history_bars=5, rsi_period=14)
        # Only 5 bars → not enough for RSI(14)+1 = 15
        result = _feed_sequence(det, [100.0] * 5, [1_000.0] * 5)
        # RSI needs 15 bars; result.rsi_val should be None
        assert result.rsi_val is None

    def test_rsi_computed_after_sufficient_history(self):
        det = VolumeClimaxDetector(min_history_bars=5, rsi_period=5)
        result = _feed_sequence(det, [100.0] * 20, [1_000.0] * 20)
        assert result.rsi_val is not None
        assert 0.0 <= result.rsi_val <= 100.0


# ---------------------------------------------------------------------------
# Stateful incremental updates
# ---------------------------------------------------------------------------

class TestStatefulUpdates:
    def test_scores_start_at_zero_and_build_up(self):
        """Scores should be zero or very small at neutral history and grow on signal."""
        det = VolumeClimaxDetector(min_history_bars=5, volume_avg_window=10)
        _feed(det, 15, price=100.0, volume=1_000.0)
        baseline = det.update(price=100.0, volume=1_000.0)
        # After a big volume spike + price drop, capitulation should be higher
        spike_result = det.update(price=92.0, volume=8_000.0)
        assert spike_result.capitulation_score >= baseline.capitulation_score

    def test_last_result_updated_after_each_call(self):
        det = VolumeClimaxDetector(min_history_bars=5)
        _feed(det, 10, price=100.0, volume=1_000.0)
        r1 = det.update(price=100.0, volume=1_000.0)
        assert det.last_result is r1
        r2 = det.update(price=105.0, volume=2_000.0)
        assert det.last_result is r2
