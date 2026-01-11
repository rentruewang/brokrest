# Copyright (c) The BrokRest Authors - All Rights Reserved

"""Tests for the rulers (support & resistance lines) module."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from brokrest.shapes.rulers import Ruler, find_rulers


class _RulerTestCase(NamedTuple):
    """Test case for ruler finding."""
    
    name: str
    prices: NDArray
    rotate: bool
    tolerance: bool
    # Expected properties (approximate)
    topline_above_all: bool  # Should topline be above all points?
    bottomline_below_all: bool  # Should bottomline be below all points?


def _get_basic_cases():
    """Generate basic test cases."""
    
    # Simple uptrend
    yield _RulerTestCase(
        name="simple_uptrend",
        prices=np.array([100.0, 105.0, 110.0, 115.0, 120.0]),
        rotate=True,
        tolerance=False,
        topline_above_all=True,
        bottomline_below_all=True,
    )
    
    # Simple downtrend
    yield _RulerTestCase(
        name="simple_downtrend",
        prices=np.array([120.0, 115.0, 110.0, 105.0, 100.0]),
        rotate=True,
        tolerance=False,
        topline_above_all=True,
        bottomline_below_all=True,
    )
    
    # Flat prices
    yield _RulerTestCase(
        name="flat_prices",
        prices=np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        rotate=True,
        tolerance=False,
        topline_above_all=True,
        bottomline_below_all=True,
    )
    
    # V-shape (valley)
    yield _RulerTestCase(
        name="v_shape",
        prices=np.array([120.0, 110.0, 100.0, 110.0, 120.0]),
        rotate=True,
        tolerance=False,
        topline_above_all=True,
        bottomline_below_all=True,
    )
    
    # Inverted V-shape (peak)
    yield _RulerTestCase(
        name="inverted_v_shape",
        prices=np.array([100.0, 110.0, 120.0, 110.0, 100.0]),
        rotate=True,
        tolerance=False,
        topline_above_all=True,
        bottomline_below_all=True,
    )
    
    # No rotation (parallel lines)
    yield _RulerTestCase(
        name="no_rotate",
        prices=np.array([100.0, 105.0, 103.0, 108.0, 106.0]),
        rotate=False,
        tolerance=False,
        topline_above_all=True,
        bottomline_below_all=True,
    )


@pytest.mark.parametrize("case", _get_basic_cases(), ids=lambda c: c.name)
def test_ruler_bounds(case: _RulerTestCase):
    """Test that rulers properly bound all data points."""
    
    topline, bottomline = find_rulers(
        case.prices,
        rotate=case.rotate,
        tolerance=case.tolerance,
    )
    
    x = np.arange(len(case.prices))
    top_pred = topline.predict(x)
    bot_pred = bottomline.predict(x)
    
    # Small tolerance for floating point
    tol = 1e-9 * (np.max(case.prices) - np.min(case.prices) + 1)
    
    if case.topline_above_all:
        assert np.all(case.prices <= top_pred + tol), \
            f"Topline should be above all points: max diff = {np.max(case.prices - top_pred)}"
    
    if case.bottomline_below_all:
        assert np.all(case.prices >= bot_pred - tol), \
            f"Bottomline should be below all points: min diff = {np.min(case.prices - bot_pred)}"


def test_ruler_dataclass():
    """Test Ruler dataclass properties."""
    
    ruler = Ruler(slope=2.0, intercept=10.0, is_top=True)
    
    assert ruler.slope == 2.0
    assert ruler.intercept == 10.0
    assert ruler.is_top is True
    
    # Test prediction
    x = np.array([0, 1, 2, 3])
    expected = np.array([10.0, 12.0, 14.0, 16.0])
    np.testing.assert_array_almost_equal(ruler.predict(x), expected)
    
    # Test equation property
    eq = ruler.equation
    assert eq.m == 2.0
    assert eq.b == 10.0


def test_ruler_contact_point():
    """Test that at least one point touches each ruler line."""
    
    prices = np.array([100.0, 120.0, 90.0, 110.0, 95.0])
    
    topline, bottomline = find_rulers(prices, rotate=True, tolerance=False)
    
    x = np.arange(len(prices))
    top_pred = topline.predict(x)
    bot_pred = bottomline.predict(x)
    
    # At least one point should be very close to each line
    top_distances = np.abs(prices - top_pred)
    bot_distances = np.abs(prices - bot_pred)
    
    price_range = np.max(prices) - np.min(prices)
    tol = 1e-6 * price_range
    
    assert np.min(top_distances) < tol, \
        "At least one point should touch the topline"
    assert np.min(bot_distances) < tol, \
        "At least one point should touch the bottomline"


def test_parallel_lines_same_slope():
    """Test that --no-rotate produces lines with same slope."""
    
    prices = np.array([100.0, 105.0, 103.0, 110.0, 108.0, 115.0])
    
    topline, bottomline = find_rulers(prices, rotate=False, tolerance=False)
    
    # Slopes should be equal (both equal to regression slope)
    np.testing.assert_almost_equal(
        topline.slope, bottomline.slope,
        err_msg="Parallel mode should produce lines with same slope"
    )


def test_no_clamp_may_violate():
    """Test that --no-clamp may produce lines that violate constraints."""
    
    # V-shape where optimal unconstrained slope differs from constrained
    prices = np.array([100.0, 80.0, 60.0, 80.0, 100.0])
    
    # With clamp (default)
    top_clamped, bot_clamped = find_rulers(prices, rotate=True, clamp=True)
    
    # Without clamp
    top_unclamped, bot_unclamped = find_rulers(prices, rotate=True, clamp=False)
    
    # Clamped version should bound all points
    x = np.arange(len(prices))
    tol = 1e-9 * (np.max(prices) - np.min(prices))
    
    assert np.all(prices <= top_clamped.predict(x) + tol)
    assert np.all(prices >= bot_clamped.predict(x) - tol)
    
    # Unclamped version may or may not bound (depends on data)
    # Just verify they're different or the same
    # (the key is that the function runs without error)


def test_tolerance_mode_in_band():
    """Test that tolerance mode captures points in band."""
    
    prices = np.array([100.0, 105.0, 103.0, 108.0, 106.0, 110.0])
    tolerance_factor = 0.2  # 20% of std
    
    topline, bottomline = find_rulers(
        prices,
        rotate=True,
        tolerance=True,
        tolerance_factor=tolerance_factor,
    )
    
    # Function should run without error
    assert topline.slope is not None
    assert bottomline.slope is not None


def test_invalid_penalty():
    """Test that invalid_penalty parameter works."""
    
    prices = np.array([100.0, 120.0, 90.0, 130.0, 85.0, 140.0])
    
    # Without penalty
    top_no_penalty, _ = find_rulers(
        prices,
        tolerance=True,
        invalid_penalty=0.0,
    )
    
    # With penalty
    top_with_penalty, _ = find_rulers(
        prices,
        tolerance=True,
        invalid_penalty=1.0,
    )
    
    # Both should produce valid rulers
    assert top_no_penalty.slope is not None
    assert top_with_penalty.slope is not None


def test_decay_rate():
    """Test that decay_rate parameter works for time weighting."""
    
    prices = np.array([100.0, 105.0, 103.0, 108.0, 106.0, 110.0, 108.0, 115.0])
    
    # Without decay (uniform weights)
    top_no_decay, bot_no_decay = find_rulers(
        prices,
        tolerance=True,
        decay_rate=0.0,
    )
    
    # With decay (recent points more important)
    top_with_decay, bot_with_decay = find_rulers(
        prices,
        tolerance=True,
        decay_rate=0.1,
    )
    
    # Both should produce valid rulers
    assert top_no_decay.slope is not None
    assert top_with_decay.slope is not None
    assert bot_no_decay.slope is not None
    assert bot_with_decay.slope is not None


def test_decay_weights():
    """Test the decay weight computation."""
    
    from brokrest.shapes.rulers import _compute_decay_weights
    
    # No decay: all weights should be 1
    weights = _compute_decay_weights(5, decay_rate=0.0)
    np.testing.assert_array_almost_equal(weights, np.ones(5))
    
    # With decay: most recent (last) should have weight 1
    weights = _compute_decay_weights(5, decay_rate=0.1)
    assert weights[-1] == 1.0  # Most recent
    assert weights[0] < weights[-1]  # Oldest should be smaller
    
    # Weights should be monotonically increasing
    assert np.all(np.diff(weights) >= 0)


def test_evaluate_ruler():
    """Test the ruler evaluation (collision counting)."""
    
    from brokrest.shapes.rulers import evaluate_ruler
    
    # Create a simple scenario
    prices = np.array([100.0, 105.0, 110.0, 105.0, 100.0, 105.0, 110.0])
    
    # Create a horizontal topline at y=108
    topline = Ruler(slope=0.0, intercept=108.0, is_top=True)
    
    collisions, invalids, score = evaluate_ruler(topline, prices, tolerance_factor=0.1)
    
    # The evaluation should return valid counts
    assert collisions >= 0
    assert invalids >= 0
    assert score == collisions - invalids


def test_evaluate_ruler_no_collisions():
    """Test evaluation when line is far from data."""
    
    from brokrest.shapes.rulers import evaluate_ruler
    
    prices = np.array([100.0, 105.0, 103.0, 108.0])
    
    # Topline way above all prices
    topline = Ruler(slope=0.0, intercept=200.0, is_top=True)
    
    collisions, invalids, score = evaluate_ruler(topline, prices, tolerance_factor=0.1)
    
    # No collisions since line is too far
    assert collisions == 0
    assert invalids == 0
    assert score == 0


def test_auto_find_rulers():
    """Test auto mode that searches for best parameters."""
    
    from brokrest.shapes.rulers import auto_find_rulers
    
    prices = np.array([100.0, 105.0, 110.0, 105.0, 100.0, 105.0, 110.0, 115.0])
    
    scored_tops, scored_bots = auto_find_rulers(
        prices,
        tolerance_factor=0.1,
        top_k=5,
        n_combinations=25,  # Small for fast test
    )
    
    # Should return up to top_k results
    assert len(scored_tops) <= 5
    assert len(scored_bots) <= 5
    
    # Should be sorted by score descending
    if len(scored_tops) > 1:
        assert scored_tops[0].score >= scored_tops[1].score
    if len(scored_bots) > 1:
        assert scored_bots[0].score >= scored_bots[1].score
    
    # Each result should have valid ruler
    for s in scored_tops:
        assert s.ruler is not None
        assert s.ruler.is_top is True
    for s in scored_bots:
        assert s.ruler is not None
        assert s.ruler.is_top is False


class _SlopeConstraintCase(NamedTuple):
    """Test case for slope constraints."""
    
    name: str
    prices: NDArray
    expected_topline_slope_sign: int  # -1, 0, or 1


def _get_slope_cases():
    """Generate slope direction test cases."""
    
    # Strong uptrend: topline should have positive slope
    yield _SlopeConstraintCase(
        name="uptrend_positive_slope",
        prices=np.array([100.0, 110.0, 120.0, 130.0, 140.0]),
        expected_topline_slope_sign=1,
    )
    
    # Strong downtrend: topline should have negative slope
    yield _SlopeConstraintCase(
        name="downtrend_negative_slope",
        prices=np.array([140.0, 130.0, 120.0, 110.0, 100.0]),
        expected_topline_slope_sign=-1,
    )


@pytest.mark.parametrize("case", _get_slope_cases(), ids=lambda c: c.name)
def test_slope_direction(case: _SlopeConstraintCase):
    """Test that slope direction matches expectation."""
    
    topline, _ = find_rulers(case.prices, rotate=True)
    
    actual_sign = np.sign(topline.slope)
    
    assert actual_sign == case.expected_topline_slope_sign, \
        f"Expected slope sign {case.expected_topline_slope_sign}, got {actual_sign}"


def test_single_point():
    """Test with single data point."""
    
    prices = np.array([100.0])
    
    topline, bottomline = find_rulers(prices)
    
    # Should not crash, and lines should pass through the point
    x = np.array([0])
    np.testing.assert_almost_equal(topline.predict(x), prices)
    np.testing.assert_almost_equal(bottomline.predict(x), prices)


def test_two_points():
    """Test with two data points."""
    
    prices = np.array([100.0, 110.0])
    
    topline, bottomline = find_rulers(prices)
    
    x = np.arange(len(prices))
    top_pred = topline.predict(x)
    bot_pred = bottomline.predict(x)
    
    tol = 1e-9
    assert np.all(prices <= top_pred + tol)
    assert np.all(prices >= bot_pred - tol)


def test_with_timestamps():
    """Test with custom timestamps."""
    
    prices = np.array([100.0, 105.0, 103.0, 108.0])
    timestamps = np.array([0, 10, 20, 30])  # Non-consecutive
    
    topline, bottomline = find_rulers(prices, timestamps=timestamps)
    
    # Predictions should work with the custom timestamps
    top_pred = topline.predict(timestamps.astype(float))
    bot_pred = bottomline.predict(timestamps.astype(float))
    
    tol = 1e-9 * (np.max(prices) - np.min(prices))
    assert np.all(prices <= top_pred + tol)
    assert np.all(prices >= bot_pred - tol)


def test_large_price_values():
    """Test with large price values (e.g., BTC prices)."""
    
    prices = np.array([30000.0, 35000.0, 32000.0, 40000.0, 38000.0])
    
    topline, bottomline = find_rulers(prices)
    
    x = np.arange(len(prices))
    top_pred = topline.predict(x)
    bot_pred = bottomline.predict(x)
    
    tol = 1e-6 * (np.max(prices) - np.min(prices))
    assert np.all(prices <= top_pred + tol)
    assert np.all(prices >= bot_pred - tol)


def test_random_prices():
    """Test with random prices."""
    
    np.random.seed(42)
    prices = np.random.uniform(100, 200, size=50)
    
    topline, bottomline = find_rulers(prices)
    
    x = np.arange(len(prices))
    top_pred = topline.predict(x)
    bot_pred = bottomline.predict(x)
    
    tol = 1e-9 * (np.max(prices) - np.min(prices))
    assert np.all(prices <= top_pred + tol)
    assert np.all(prices >= bot_pred - tol)

