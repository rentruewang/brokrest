# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Support and resistance line detection (Rulers).

Finds the optimal topline (resistance) and bottomline (support) that bound all data points.
"""

from __future__ import annotations

import dataclasses as dcls
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from sklearn.linear_model import LinearRegression

from brokrest.equations import SlopeInterceptForm
from brokrest.painters import Canvas, Painter
from brokrest.vectors import Vec2d

if TYPE_CHECKING:
    from typing import Optional

__all__ = ["Ruler", "find_rulers", "evaluate_ruler", "auto_find_rulers"]


@dcls.dataclass(frozen=True)
class Ruler(Painter):
    """
    A bounding line (support or resistance) for price data.
    
    The line is defined as y = slope * x + intercept.
    """

    slope: float
    intercept: float
    is_top: bool  # True for resistance (topline), False for support (bottomline)

    @property
    def equation(self) -> SlopeInterceptForm:
        """Linear equation: y = mx + b"""
        return SlopeInterceptForm(m=self.slope, b=self.intercept)

    def predict(self, x: NDArray) -> NDArray:
        """Predict y values for given x positions."""
        return self.slope * x + self.intercept

    def plot(self, canvas: Canvas) -> None:
        """Draw ruler line on matplotlib canvas."""
        xs = canvas.xs()
        y_start = self.predict(np.array([xs[0]]))[0]
        y_end = self.predict(np.array([xs[-1]]))[0]
        
        color = "purple" if self.is_top else "orange"
        canvas.line(Vec2d(xs[0], y_start), Vec2d(xs[-1], y_end), color=color)


def _linear_regression(x: NDArray, y: NDArray) -> Tuple[float, float]:
    """Simple linear regression, returns (slope, intercept)."""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0]), float(model.intercept_)


def evaluate_ruler(
    ruler: Ruler,
    prices: NDArray,
    tolerance_factor: float = 0.1,
) -> Tuple[int, int, int]:
    """
    Evaluate a ruler line by counting "collisions" with the price data.
    
    A collision is when the price sequence:
    1. Was outside the tolerance band
    2. Enters the tolerance band
    3. Later exits the tolerance band
    
    Multiple consecutive points inside (or outside) the band count as one event.
    
    For topline: "invalid" means price is above the line (beyond tolerance).
    For bottomline: "invalid" means price is below the line (beyond tolerance).
    
    Args:
        ruler: The Ruler to evaluate
        prices: Price array
        tolerance_factor: Band size as fraction of price std
        
    Returns:
        (collision_count, invalid_count, score) where score = collisions - invalids
    """
    n = len(prices)
    x = np.arange(n, dtype=float)
    pred = ruler.predict(x)
    band = tolerance_factor * np.std(prices)
    
    if band < 1e-10:
        return 0, 0, 0
    
    # Calculate distance from line
    # For topline: positive = below line (valid side), negative = above line (invalid side)
    # For bottomline: positive = above line (valid side), negative = below line (invalid side)
    if ruler.is_top:
        # Topline: points should be below (pred - prices > 0 means below)
        distances = pred - prices
    else:
        # Bottomline: points should be above (prices - pred > 0 means above)
        distances = prices - pred
    
    # Classify each point:
    # in_band: |distance| <= band (within tolerance)
    # valid_outside: distance > band (on the correct side, outside tolerance)
    # invalid: distance < -band (on the wrong side, beyond tolerance)
    in_band = np.abs(distances) <= band
    invalid = distances < -band
    valid_outside = distances > band
    
    # Count collisions: transitions from outside -> in_band
    # Count invalid runs: sequences of consecutive invalid points
    
    collision_count = 0
    invalid_run_count = 0
    
    prev_in_band = False
    prev_invalid = False
    
    for i in range(n):
        curr_in_band = in_band[i]
        curr_invalid = invalid[i]
        
        # Collision: was not in band, now in band
        if curr_in_band and not prev_in_band:
            collision_count += 1
        
        # New invalid run: was not invalid, now invalid
        if curr_invalid and not prev_invalid:
            invalid_run_count += 1
        
        prev_in_band = curr_in_band
        prev_invalid = curr_invalid
    
    score = collision_count - invalid_run_count
    return collision_count, invalid_run_count, score


def _compute_decay_weights(n: int, decay_rate: float) -> NDArray:
    """
    Compute exponential decay weights for time series.
    
    Recent points (higher index) get higher weights.
    weights[i] = exp(-decay_rate * (n - 1 - i))
    
    Args:
        n: Number of data points
        decay_rate: Decay rate. 0.0 = uniform weights, higher = more emphasis on recent.
        
    Returns:
        Array of weights, normalized so that the most recent point has weight 1.0
    """
    if decay_rate <= 0:
        return np.ones(n)
    
    # Distance from the most recent point (index n-1)
    # Most recent point has distance 0, oldest has distance n-1
    distances = np.arange(n - 1, -1, -1)  # [n-1, n-2, ..., 1, 0]
    weights = np.exp(-decay_rate * distances)
    return weights


def _optimize_topline_with_band(
    x: NDArray,
    y: NDArray,
    m: float,
    b: float,
    tolerance_factor: float = 0.1,
    invalid_penalty: float = 0.0,
    decay_rate: float = 0.0,
) -> Tuple[float, float]:
    """
    Optimize topline by moving it inward to capture more points within a band.
    
    If only one point touches the strict topline, the line may not be meaningful.
    This function tries to move the line down (inward) so that more points fall
    within a tolerance band around the line.
    
    Args:
        x, y: Data arrays
        m, b: Current topline (y = mx + b)
        tolerance_factor: Band size as fraction of price std
        invalid_penalty: Penalty coefficient for points above the line (violated).
                         0.0 = ignore violations, 0.5 = moderate penalty, 1.0 = full penalty.
        decay_rate: Exponential decay rate for time weighting.
                    0.0 = all points equally important (default).
                    Higher values = recent points more important.
        
    Returns:
        (optimized_slope, optimized_intercept)
    """
    band = tolerance_factor * np.std(y)
    if band < 1e-10:
        return m, b
    
    n = len(x)
    weights = _compute_decay_weights(n, decay_rate)
    
    # Current line prediction
    pred = m * x + b
    
    # For topline: distance = pred - y (positive means point is below line)
    distances = pred - y
    
    # Current: weighted count of points in band
    in_band_mask = np.abs(distances) <= band
    in_band_current = np.sum(weights[in_band_mask])
    
    # Invalid points: above the line (distance < 0)
    invalid_mask = distances < 0
    invalid_current = np.sum(weights[invalid_mask])
    
    current_score = in_band_current - invalid_penalty * invalid_current
    
    # If already have many points in band, no need to optimize
    if np.sum(in_band_mask) >= n * 0.3:
        return m, b
    
    # Try shifting the line down to where more points cluster
    # Key idea: find the shift that maximizes weighted (in_band - invalid_penalty * invalid)
    
    best_b = b
    best_score = current_score
    
    # Try different shift amounts: target each point to be AT the line
    for target_idx in range(n):
        # Shift so this point is exactly on the line
        new_b = y[target_idx] - m * x[target_idx]
        
        new_pred = m * x + new_b
        new_distances = new_pred - y
        
        # Weighted count of points in band
        in_band_mask = np.abs(new_distances) <= band
        in_band = np.sum(weights[in_band_mask])
        
        # Weighted count of invalid points
        invalid_mask = new_distances < 0
        invalid = np.sum(weights[invalid_mask])
        
        # Score: maximize weighted in_band - invalid_penalty * weighted invalid
        score = in_band - invalid_penalty * invalid
        
        if score > best_score:
            best_score = score
            best_b = new_b
    
    return m, best_b


def _find_topline(
    x: NDArray, 
    y: NDArray, 
    tolerance: bool = False,
    tolerance_factor: float = 0.1,
    clamp: bool = True,
    invalid_penalty: float = 0.0,
    decay_rate: float = 0.0,
) -> Tuple[float, float]:
    """
    Find the optimal topline (resistance line).
    
    Algorithm:
    1. Start with linear regression line
    2. Shift up until all points are below
    3. Pivot on contact point and optimize slope to minimize MSE
       while keeping all points below the line (if clamp=True)
    
    Args:
        x, y: Data arrays
        tolerance: If True, allow the line to move inward to capture more points
                   within a tolerance band, rather than strictly touching extreme points.
        tolerance_factor: Band size as fraction of price std (default 0.1 = 10%)
        clamp: If True, clamp slope to feasible range. If False, use unconstrained optimal.
        invalid_penalty: Penalty for points above the line in tolerance mode (0.0-1.0).
        decay_rate: Exponential decay for time weighting. Recent points more important.
    
    Returns:
        (slope, intercept)
    """
    n = len(x)
    
    # Step 1: Linear regression as baseline
    m_base, b_base = _linear_regression(x, y)
    
    # Step 2: Shift up until all points are below or on the line
    residuals = y - (m_base * x + b_base)
    delta = np.max(residuals)
    b_shifted = b_base + delta
    
    # Step 3: Find contact point (point touching the line)
    pred = m_base * x + b_shifted
    tol = 1e-9 * (np.max(y) - np.min(y))
    contact_mask = np.abs(y - pred) < tol
    
    if not np.any(contact_mask):
        # No exact contact, find the closest
        contact_idx = np.argmax(y - (m_base * x + b_base))
    else:
        contact_idx = np.where(contact_mask)[0][0]
    
    pivot_x = x[contact_idx]
    pivot_y = y[contact_idx]
    
    # Step 4: Optimize slope with pivot constraint
    # Line: y = m * (x - pivot_x) + pivot_y
    # Constraint: y_i <= m * (x_i - pivot_x) + pivot_y for all i
    # Objective: minimize MSE
    
    dx = x - pivot_x
    dy = y - pivot_y
    
    # Compute slope bounds from constraints
    # For x_i > pivot_x (dx > 0): m >= dy_i / dx_i
    # For x_i < pivot_x (dx < 0): m <= dy_i / dx_i
    
    right_mask = dx > tol
    left_mask = dx < -tol
    
    m_lower = -np.inf
    m_upper = np.inf
    right_constraint_idx = None
    left_constraint_idx = None
    
    if np.any(right_mask):
        slopes_right = dy[right_mask] / dx[right_mask]
        m_lower = np.max(slopes_right)
        # Find which point gives the tightest constraint
        right_indices = np.where(right_mask)[0]
        right_constraint_idx = right_indices[np.argmax(slopes_right)]
    
    if np.any(left_mask):
        slopes_left = dy[left_mask] / dx[left_mask]
        m_upper = np.min(slopes_left)
        # Find which point gives the tightest constraint
        left_indices = np.where(left_mask)[0]
        left_constraint_idx = left_indices[np.argmin(slopes_left)]
    
    if m_lower > m_upper:
        # Constraints conflict - pivot is a convex point
        # Fall back to convex hull based approach
        return _find_topline_convex(x, y)
    
    # Find optimal slope in [m_lower, m_upper] that minimizes MSE
    # The MSE is a quadratic in m, so we can find the minimum analytically
    # m_opt = mean(dy*dx) / mean(dx^2)
    
    mean_dx2 = np.mean(dx ** 2)
    if mean_dx2 > tol:
        m_unconstrained = np.mean(dy * dx) / mean_dx2
    else:
        m_unconstrained = m_base
    
    # Clamp to feasible range (or use unconstrained if clamp=False)
    if clamp:
        m_opt = np.clip(m_unconstrained, m_lower, m_upper)
    else:
        m_opt = m_unconstrained
    b_opt = pivot_y - m_opt * pivot_x
    
    # Tolerance mode: if only one contact point, try to move line inward
    # to capture more points within a tolerance band
    if tolerance:
        m_opt, b_opt = _optimize_topline_with_band(
            x, y, m_opt, b_opt, tolerance_factor, invalid_penalty, decay_rate
        )
    
    return m_opt, b_opt


def _find_topline_convex(x: NDArray, y: NDArray) -> Tuple[float, float]:
    """
    Fallback: Find topline using convex hull approach.
    
    Finds the line segment on the upper convex hull that minimizes MSE.
    """
    from scipy.spatial import ConvexHull
    
    points = np.column_stack([x, y])
    
    if len(points) < 3:
        return _linear_regression(x, y)
    
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Find upper hull (points where we go from left to right)
    sorted_idx = np.argsort(hull_points[:, 0])
    upper_points = hull_points[sorted_idx]
    
    best_m, best_b = None, None
    best_mse = np.inf
    
    for i in range(len(upper_points) - 1):
        x1, y1 = upper_points[i]
        x2, y2 = upper_points[i + 1]
        
        if abs(x2 - x1) < 1e-10:
            continue
        
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        
        # Check if all points are below or on the line
        pred = m * x + b
        if np.all(y <= pred + 1e-9 * (np.max(y) - np.min(y))):
            mse = np.mean((y - pred) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_m, best_b = m, b
    
    if best_m is None:
        # Fallback to shifted regression
        m, b = _linear_regression(x, y)
        delta = np.max(y - (m * x + b))
        return m, b + delta
    
    return best_m, best_b


def _find_bottomline(
    x: NDArray, 
    y: NDArray, 
    tolerance: bool = False,
    tolerance_factor: float = 0.1,
    clamp: bool = True,
    invalid_penalty: float = 0.0,
    decay_rate: float = 0.0,
) -> Tuple[float, float]:
    """
    Find the optimal bottomline (support line).
    
    Algorithm:
    1. Start with linear regression line
    2. Shift down until all points are above
    3. Pivot on contact point and optimize slope to minimize MSE
       while keeping all points above the line (if clamp=True)
    
    Args:
        x, y: Data arrays
        tolerance: If True, when constraints conflict, connect the two
                   tightest constraint points directly (may have some violations)
        clamp: If True, clamp slope to feasible range. If False, use unconstrained optimal.
        invalid_penalty: Penalty for points below the line in tolerance mode (0.0-1.0).
        decay_rate: Exponential decay for time weighting. Recent points more important.
    
    Returns:
        (slope, intercept)
    """
    n = len(x)
    
    # Step 1: Linear regression as baseline
    m_base, b_base = _linear_regression(x, y)
    
    # Step 2: Shift down until all points are above or on the line
    residuals = y - (m_base * x + b_base)
    delta = np.min(residuals)  # This is negative or zero
    b_shifted = b_base + delta
    
    # Step 3: Find contact point (point touching the line)
    pred = m_base * x + b_shifted
    tol = 1e-9 * (np.max(y) - np.min(y))
    contact_mask = np.abs(y - pred) < tol
    
    if not np.any(contact_mask):
        # No exact contact, find the closest
        contact_idx = np.argmin(y - (m_base * x + b_base))
    else:
        contact_idx = np.where(contact_mask)[0][0]
    
    pivot_x = x[contact_idx]
    pivot_y = y[contact_idx]
    
    # Step 4: Optimize slope with pivot constraint
    # Line: y = m * (x - pivot_x) + pivot_y
    # Constraint: y_i >= m * (x_i - pivot_x) + pivot_y for all i (opposite of topline)
    # Objective: minimize MSE
    
    dx = x - pivot_x
    dy = y - pivot_y
    
    # Compute slope bounds from constraints
    # For x_i > pivot_x (dx > 0): m <= dy_i / dx_i
    # For x_i < pivot_x (dx < 0): m >= dy_i / dx_i
    
    right_mask = dx > tol
    left_mask = dx < -tol
    
    m_lower = -np.inf
    m_upper = np.inf
    right_constraint_idx = None
    left_constraint_idx = None
    
    if np.any(right_mask):
        slopes_right = dy[right_mask] / dx[right_mask]
        m_upper = np.min(slopes_right)  # Opposite of topline
        right_indices = np.where(right_mask)[0]
        right_constraint_idx = right_indices[np.argmin(slopes_right)]
    
    if np.any(left_mask):
        slopes_left = dy[left_mask] / dx[left_mask]
        m_lower = np.max(slopes_left)  # Opposite of topline
        left_indices = np.where(left_mask)[0]
        left_constraint_idx = left_indices[np.argmax(slopes_left)]
    
    if m_lower > m_upper:
        # Constraints conflict - fallback to simple shifted line
        return m_base, b_shifted
    
    # Find optimal slope in [m_lower, m_upper] that minimizes MSE
    mean_dx2 = np.mean(dx ** 2)
    if mean_dx2 > tol:
        m_unconstrained = np.mean(dy * dx) / mean_dx2
    else:
        m_unconstrained = m_base
    
    # Clamp to feasible range (or use unconstrained if clamp=False)
    if clamp:
        m_opt = np.clip(m_unconstrained, m_lower, m_upper)
    else:
        m_opt = m_unconstrained
    b_opt = pivot_y - m_opt * pivot_x
    
    # Tolerance mode: try to move line inward to capture more points in band
    if tolerance:
        m_opt, b_opt = _optimize_bottomline_with_band(
            x, y, m_opt, b_opt, tolerance_factor, invalid_penalty, decay_rate
        )
    
    return m_opt, b_opt


def _optimize_bottomline_with_band(
    x: NDArray,
    y: NDArray,
    m: float,
    b: float,
    tolerance_factor: float = 0.1,
    invalid_penalty: float = 0.0,
    decay_rate: float = 0.0,
) -> Tuple[float, float]:
    """
    Optimize bottomline by moving it inward to capture more points within a band.
    
    Similar to _optimize_topline_with_band but for the support line.
    
    Args:
        x, y: Data arrays
        m, b: Current bottomline (y = mx + b)
        tolerance_factor: Band size as fraction of price std
        invalid_penalty: Penalty coefficient for points below the line (violated).
                         0.0 = ignore violations, 0.5 = moderate penalty, 1.0 = full penalty.
        decay_rate: Exponential decay rate for time weighting.
                    0.0 = all points equally important (default).
                    Higher values = recent points more important.
    """
    band = tolerance_factor * np.std(y)
    if band < 1e-10:
        return m, b
    
    n = len(x)
    weights = _compute_decay_weights(n, decay_rate)
    
    # Current line prediction
    pred = m * x + b
    
    # For bottomline: distance = y - pred (positive means point is above line)
    distances = y - pred
    
    # Current: weighted count of points in band
    in_band_mask = np.abs(distances) <= band
    in_band_current = np.sum(weights[in_band_mask])
    
    # Invalid points: below the line (distance < 0)
    invalid_mask = distances < 0
    invalid_current = np.sum(weights[invalid_mask])
    
    current_score = in_band_current - invalid_penalty * invalid_current
    
    # If already have many points in band, no need to optimize
    if np.sum(in_band_mask) >= n * 0.3:
        return m, b
    
    best_b = b
    best_score = current_score
    
    # Try different shift amounts: target each point to be AT the line
    for target_idx in range(n):
        new_b = y[target_idx] - m * x[target_idx]
        
        new_pred = m * x + new_b
        new_distances = y - new_pred
        
        # Weighted count of points in band
        in_band_mask = np.abs(new_distances) <= band
        in_band = np.sum(weights[in_band_mask])
        
        # Weighted count of invalid points
        invalid_mask = new_distances < 0
        invalid = np.sum(weights[invalid_mask])
        
        # Score: maximize weighted in_band - invalid_penalty * weighted invalid
        score = in_band - invalid_penalty * invalid
        
        if score > best_score:
            best_score = score
            best_b = new_b
    
    return m, best_b


def find_rulers(
    prices: NDArray,
    timestamps: Optional[NDArray] = None,
    rotate: bool = True,
    tolerance: bool = False,
    tolerance_factor: float = 0.1,
    clamp: bool = True,
    invalid_penalty: float = 0.0,
    decay_rate: float = 0.0,
) -> Tuple[Ruler, Ruler]:
    """
    Find support and resistance lines for price data.
    
    Args:
        prices: Price array
        timestamps: Optional timestamp array (uses indices if None)
        rotate: If True, optimize slope after shifting. If False, keep parallel to regression line.
        tolerance: If True, allow lines to move inward to capture more points within
                   a tolerance band, rather than strictly touching extreme points.
        tolerance_factor: Band size as fraction of price std (default 0.1 = 10%)
        clamp: If True, clamp slope to feasible range. If False, use unconstrained optimal
               (pure linear regression on pivot-centered data).
        invalid_penalty: Penalty coefficient for points that violate the line constraint
                         in tolerance mode. 0.0 = ignore violations (maximize in-band only),
                         0.5 = moderate penalty, 1.0 = full penalty.
        decay_rate: Exponential decay rate for time weighting in tolerance mode.
                    0.0 = all points equally important (default).
                    Higher values = recent points more important.
                    Example: 0.01 means weight halves every ~70 time steps.
        
    Returns:
        (topline, bottomline) - Ruler objects for resistance and support
    """
    if timestamps is None:
        x = np.arange(len(prices), dtype=float)
    else:
        x = timestamps.astype(float)
    
    y = np.asarray(prices, dtype=float)
    
    if rotate:
        # Find topline (resistance) with slope optimization
        m_top, b_top = _find_topline(
            x, y, tolerance=tolerance, tolerance_factor=tolerance_factor, 
            clamp=clamp, invalid_penalty=invalid_penalty, decay_rate=decay_rate
        )
        # Find bottomline (support) with slope optimization
        m_bot, b_bot = _find_bottomline(
            x, y, tolerance=tolerance, tolerance_factor=tolerance_factor, 
            clamp=clamp, invalid_penalty=invalid_penalty, decay_rate=decay_rate
        )
    else:
        # Parallel lines: only shift, no rotation
        m_top, b_top = _find_topline_parallel(x, y)
        m_bot, b_bot = _find_bottomline_parallel(x, y)
    
    topline = Ruler(slope=m_top, intercept=b_top, is_top=True)
    bottomline = Ruler(slope=m_bot, intercept=b_bot, is_top=False)
    
    return topline, bottomline


@dcls.dataclass
class ScoredRuler:
    """A ruler with its evaluation score and parameters."""
    ruler: Ruler
    score: int
    collisions: int
    invalids: int
    invalid_penalty: float
    decay_rate: float


def auto_find_rulers(
    prices: NDArray,
    tolerance_factor: float = 0.1,
    top_k: int = 10,
    n_combinations: int = 100,
) -> Tuple[list, list]:
    """
    Automatically find the best rulers by testing different parameter combinations.
    
    Tests various combinations of invalid_penalty and decay_rate, then returns
    the top-scoring rulers for both topline and bottomline.
    
    Args:
        prices: Price array
        tolerance_factor: Band size as fraction of price std
        top_k: Number of top-scoring rulers to return (default 10)
        n_combinations: Approximate number of parameter combinations to try
        
    Returns:
        (top_scored_toplines, top_scored_bottomlines) - Lists of ScoredRuler objects
    """
    # Generate parameter combinations
    # invalid_penalty: 0.0 to 1.0
    # decay_rate: 0.0 to 0.1
    
    n_penalty = int(np.sqrt(n_combinations))
    n_decay = n_combinations // n_penalty
    
    penalty_values = np.linspace(0.0, 1.0, n_penalty)
    decay_values = np.linspace(0.0, 0.1, n_decay)
    
    topline_results = []
    bottomline_results = []
    
    for penalty in penalty_values:
        for decay in decay_values:
            topline, bottomline = find_rulers(
                prices,
                rotate=True,
                tolerance=True,
                tolerance_factor=tolerance_factor,
                clamp=True,
                invalid_penalty=penalty,
                decay_rate=decay,
            )
            
            # Evaluate
            top_coll, top_inv, top_score = evaluate_ruler(topline, prices, tolerance_factor)
            bot_coll, bot_inv, bot_score = evaluate_ruler(bottomline, prices, tolerance_factor)
            
            topline_results.append(ScoredRuler(
                ruler=topline,
                score=top_score,
                collisions=top_coll,
                invalids=top_inv,
                invalid_penalty=penalty,
                decay_rate=decay,
            ))
            
            bottomline_results.append(ScoredRuler(
                ruler=bottomline,
                score=bot_score,
                collisions=bot_coll,
                invalids=bot_inv,
                invalid_penalty=penalty,
                decay_rate=decay,
            ))
    
    # Sort by score (descending) and take top_k
    topline_results.sort(key=lambda x: x.score, reverse=True)
    bottomline_results.sort(key=lambda x: x.score, reverse=True)
    
    # Remove duplicates (same slope and intercept)
    def dedupe(results):
        seen = set()
        deduped = []
        for r in results:
            key = (round(r.ruler.slope, 4), round(r.ruler.intercept, 2))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
            if len(deduped) >= top_k:
                break
        return deduped
    
    return dedupe(topline_results), dedupe(bottomline_results)


def _find_topline_parallel(x: NDArray, y: NDArray) -> Tuple[float, float]:
    """
    Find topline by shifting regression line up (no rotation).
    """
    m, b = _linear_regression(x, y)
    residuals = y - (m * x + b)
    delta = np.max(residuals)
    return m, b + delta


def _find_bottomline_parallel(x: NDArray, y: NDArray) -> Tuple[float, float]:
    """
    Find bottomline by shifting regression line down (no rotation).
    """
    m, b = _linear_regression(x, y)
    residuals = y - (m * x + b)
    delta = np.min(residuals)
    return m, b + delta


def plot_rulers_mpl(
    prices: NDArray,
    topline: Ruler,
    bottomline: Ruler,
    regression_line: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Support & Resistance Lines",
    tolerance_factor: Optional[float] = None,
):
    """
    Plot prices with support and resistance lines using matplotlib.
    
    Args:
        prices: Price array
        topline: Resistance line
        bottomline: Support line
        regression_line: Whether to show the base regression line
        figsize: Figure size
        title: Plot title
        tolerance_factor: If set, color points in-band with line color, others green
        
    Returns:
        Matplotlib figure
    """
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    
    x = np.arange(len(prices))
    y_top = topline.predict(x)
    y_bot = bottomline.predict(x)
    
    # Plot price line
    ax.plot(x, prices, color="gray", alpha=0.5, linewidth=1, label="Price")
    
    # Color points based on tolerance mode
    if tolerance_factor is not None:
        band = tolerance_factor * np.std(prices)
        
        # Distance from topline (positive = below line)
        dist_top = y_top - prices
        in_band_top = np.abs(dist_top) <= band
        
        # Distance from bottomline (positive = above line)
        dist_bot = prices - y_bot
        in_band_bot = np.abs(dist_bot) <= band
        
        # Points in top band: red
        # Points in bottom band: blue  
        # Points in neither: green
        in_neither = ~in_band_top & ~in_band_bot
        
        if np.any(in_band_top):
            ax.scatter(x[in_band_top], prices[in_band_top], color="red", s=15, 
                       alpha=0.7, label=f"Near resistance ({np.sum(in_band_top)})")
        if np.any(in_band_bot):
            ax.scatter(x[in_band_bot], prices[in_band_bot], color="blue", s=15,
                       alpha=0.7, label=f"Near support ({np.sum(in_band_bot)})")
        if np.any(in_neither):
            ax.scatter(x[in_neither], prices[in_neither], color="green", s=10,
                       alpha=0.5, label=f"Other ({np.sum(in_neither)})")
    else:
        ax.scatter(x, prices, color="gray", s=10, alpha=0.5)
    
    # Plot regression line
    if regression_line:
        m, b = _linear_regression(x, prices)
        ax.plot(x, m * x + b, color="gray", linestyle=":", linewidth=1.5, 
                label=f"Regression")
    
    # Evaluate rulers and compute scores
    tol_factor = tolerance_factor if tolerance_factor is not None else 0.1
    top_collisions, top_invalids, top_score = evaluate_ruler(topline, prices, tol_factor)
    bot_collisions, bot_invalids, bot_score = evaluate_ruler(bottomline, prices, tol_factor)
    
    # Plot topline (resistance) with score annotation
    ax.plot(x, y_top, color="red", linestyle="--", linewidth=2,
            label=f"Resistance (y={topline.slope:.4f}x+{topline.intercept:.2f})")
    
    # Annotate topline score at the right end
    top_score_text = f"Score: {top_score} ({top_collisions}↑ - {top_invalids}✗)"
    ax.annotate(top_score_text, 
                xy=(x[-1], y_top[-1]), 
                xytext=(5, 10), textcoords='offset points',
                fontsize=9, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))
    
    # Plot bottomline (support) with score annotation
    ax.plot(x, y_bot, color="blue", linestyle="--", linewidth=2,
            label=f"Support (y={bottomline.slope:.4f}x+{bottomline.intercept:.2f})")
    
    # Annotate bottomline score at the right end
    bot_score_text = f"Score: {bot_score} ({bot_collisions}↑ - {bot_invalids}✗)"
    ax.annotate(bot_score_text,
                xy=(x[-1], y_bot[-1]),
                xytext=(5, -15), textcoords='offset points',
                fontsize=9, color="blue", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8))
    
    # Legend outside plot on the right
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Make room for legend on right
    
    return fig


def plot_rulers_bokeh(
    prices: NDArray,
    topline: Ruler,
    bottomline: Ruler,
    regression_line: bool = True,
    width: int = 1200,
    height: int = 600,
    title: str = "Support & Resistance Lines",
    tolerance_factor: Optional[float] = None,
):
    """
    Plot prices with support and resistance lines using Bokeh.
    
    Args:
        prices: Price array
        topline: Resistance line
        bottomline: Support line
        regression_line: Whether to show the base regression line
        width: Plot width
        height: Plot height
        title: Plot title
        tolerance_factor: If set, color points in-band with line color, others green
    
    Returns:
        Bokeh figure
    """
    from bokeh.plotting import figure
    
    x = np.arange(len(prices))
    y_top = topline.predict(x)
    y_bot = bottomline.predict(x)
    
    fig = figure(
        title=title,
        x_axis_label="Time",
        y_axis_label="Price",
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    
    # Plot price line
    fig.line(x=x, y=prices, color="gray", alpha=0.5, line_width=1, legend_label="Price")
    
    # Color points based on tolerance mode
    if tolerance_factor is not None:
        band = tolerance_factor * np.std(prices)
        
        # Distance from topline
        dist_top = y_top - prices
        in_band_top = np.abs(dist_top) <= band
        
        # Distance from bottomline
        dist_bot = prices - y_bot
        in_band_bot = np.abs(dist_bot) <= band
        
        # Points in neither band
        in_neither = ~in_band_top & ~in_band_bot
        
        if np.any(in_band_top):
            fig.scatter(x=x[in_band_top], y=prices[in_band_top], color="red", 
                        size=8, alpha=0.7, legend_label=f"Near resistance ({np.sum(in_band_top)})")
        if np.any(in_band_bot):
            fig.scatter(x=x[in_band_bot], y=prices[in_band_bot], color="blue",
                        size=8, alpha=0.7, legend_label=f"Near support ({np.sum(in_band_bot)})")
        if np.any(in_neither):
            fig.scatter(x=x[in_neither], y=prices[in_neither], color="green",
                        size=5, alpha=0.5, legend_label=f"Other ({np.sum(in_neither)})")
    else:
        fig.scatter(x=x, y=prices, color="gray", size=5, alpha=0.5)
    
    # Plot regression line
    if regression_line:
        m, b = _linear_regression(x, prices)
        fig.line(x=x, y=m * x + b, color="gray", line_dash="dotted", 
                 line_width=1.5, legend_label="Regression")
    
    # Evaluate rulers and compute scores
    tol_factor = tolerance_factor if tolerance_factor is not None else 0.1
    top_collisions, top_invalids, top_score = evaluate_ruler(topline, prices, tol_factor)
    bot_collisions, bot_invalids, bot_score = evaluate_ruler(bottomline, prices, tol_factor)
    
    # Plot topline (resistance) with score in legend
    top_label = f"Resistance [Score: {top_score} ({top_collisions}↑-{top_invalids}✗)]"
    fig.line(x=x, y=y_top, color="red", line_dash="dashed", line_width=2,
             legend_label=top_label)
    
    # Plot bottomline (support) with score in legend
    bot_label = f"Support [Score: {bot_score} ({bot_collisions}↑-{bot_invalids}✗)]"
    fig.line(x=x, y=y_bot, color="blue", line_dash="dashed", line_width=2,
             legend_label=bot_label)
    
    # Add text annotations for scores at the end of lines
    from bokeh.models import Label
    
    top_score_label = Label(
        x=x[-1], y=y_top[-1],
        text=f" Score: {top_score}",
        text_font_size="10pt", text_color="red", text_font_style="bold",
        x_offset=5, y_offset=5
    )
    fig.add_layout(top_score_label)
    
    bot_score_label = Label(
        x=x[-1], y=y_bot[-1],
        text=f" Score: {bot_score}",
        text_font_size="10pt", text_color="blue", text_font_style="bold",
        x_offset=5, y_offset=-15
    )
    fig.add_layout(bot_score_label)
    
    # Move legend outside to the right
    fig.add_layout(fig.legend[0], 'right')
    fig.legend.click_policy = "hide"
    
    return fig


def plot_rulers_auto_mpl(
    prices: NDArray,
    scored_toplines: list,
    scored_bottomlines: list,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Auto-discovered Support & Resistance Lines",
    tolerance_factor: float = 0.1,
):
    """
    Plot multiple auto-discovered rulers using matplotlib.
    
    Args:
        prices: Price array
        scored_toplines: List of ScoredRuler objects for toplines
        scored_bottomlines: List of ScoredRuler objects for bottomlines
        figsize: Figure size
        title: Plot title
        tolerance_factor: Band size for coloring
        
    Returns:
        Matplotlib figure
    """
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    
    x = np.arange(len(prices))
    
    # Plot price data
    ax.plot(x, prices, color="gray", alpha=0.6, linewidth=1, label="Price")
    ax.scatter(x, prices, color="gray", s=5, alpha=0.3)
    
    # Color maps for toplines (reds) and bottomlines (blues)
    n_top = len(scored_toplines)
    n_bot = len(scored_bottomlines)
    
    top_colors = cm.Reds(np.linspace(0.4, 0.9, max(n_top, 1)))
    bot_colors = cm.Blues(np.linspace(0.4, 0.9, max(n_bot, 1)))
    
    # Plot toplines (resistance)
    for i, scored in enumerate(scored_toplines):
        ruler = scored.ruler
        y_pred = ruler.predict(x)
        alpha = 0.9 - i * 0.05  # Fade out lower-ranked lines
        lw = 2.5 - i * 0.15
        
        ax.plot(x, y_pred, color=top_colors[i], linestyle="--", linewidth=lw, alpha=alpha)
        
        # Annotate score at right end
        label = f"{scored.score}"
        ax.annotate(label,
                    xy=(x[-1], y_pred[-1]),
                    xytext=(3, 0), textcoords='offset points',
                    fontsize=8, color=top_colors[i], fontweight="bold",
                    alpha=alpha)
    
    # Plot bottomlines (support)
    for i, scored in enumerate(scored_bottomlines):
        ruler = scored.ruler
        y_pred = ruler.predict(x)
        alpha = 0.9 - i * 0.05
        lw = 2.5 - i * 0.15
        
        ax.plot(x, y_pred, color=bot_colors[i], linestyle="--", linewidth=lw, alpha=alpha)
        
        # Annotate score at right end
        label = f"{scored.score}"
        ax.annotate(label,
                    xy=(x[-1], y_pred[-1]),
                    xytext=(3, 0), textcoords='offset points',
                    fontsize=8, color=bot_colors[i], fontweight="bold",
                    alpha=alpha)
    
    # Create legend entries for top lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', label='Price'),
    ]
    
    if n_top > 0:
        best_top = scored_toplines[0]
        legend_elements.append(
            Line2D([0], [0], color=top_colors[0], linestyle='--', linewidth=2,
                   label=f'Best Resistance: {best_top.score} (p={best_top.invalid_penalty:.2f}, d={best_top.decay_rate:.3f})')
        )
    
    if n_bot > 0:
        best_bot = scored_bottomlines[0]
        legend_elements.append(
            Line2D([0], [0], color=bot_colors[0], linestyle='--', linewidth=2,
                   label=f'Best Support: {best_bot.score} (p={best_bot.invalid_penalty:.2f}, d={best_bot.decay_rate:.3f})')
        )
    
    # Add summary
    if n_top > 0:
        legend_elements.append(
            Line2D([0], [0], color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Top {n_top} Resistance lines')
        )
    if n_bot > 0:
        legend_elements.append(
            Line2D([0], [0], color='blue', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Top {n_bot} Support lines')
        )
    
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), 
              fontsize=8, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    
    return fig


def plot_rulers_auto_bokeh(
    prices: NDArray,
    scored_toplines: list,
    scored_bottomlines: list,
    width: int = 1400,
    height: int = 700,
    title: str = "Auto-discovered Support & Resistance Lines",
    tolerance_factor: float = 0.1,
):
    """
    Plot multiple auto-discovered rulers using Bokeh.
    
    Args:
        prices: Price array
        scored_toplines: List of ScoredRuler objects for toplines
        scored_bottomlines: List of ScoredRuler objects for bottomlines
        width: Plot width
        height: Plot height
        title: Plot title
        tolerance_factor: Band size for coloring
        
    Returns:
        Bokeh figure
    """
    from bokeh.plotting import figure
    from bokeh.models import Label
    
    x = np.arange(len(prices))
    
    fig = figure(
        title=title,
        x_axis_label="Time",
        y_axis_label="Price",
        width=width,
        height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    
    # Plot price data
    fig.line(x=x, y=prices, color="gray", alpha=0.6, line_width=1, legend_label="Price")
    fig.scatter(x=x, y=prices, color="gray", size=3, alpha=0.3)
    
    # Get color palettes
    n_top = len(scored_toplines)
    n_bot = len(scored_bottomlines)
    
    # Generate colors manually (darker = higher score)
    def generate_reds(n):
        if n == 0:
            return []
        # From dark red to light red
        return [f"rgb({180 + int(75 * i / max(n-1, 1))}, {50 + int(100 * i / max(n-1, 1))}, {50 + int(100 * i / max(n-1, 1))})" for i in range(n)]
    
    def generate_blues(n):
        if n == 0:
            return []
        # From dark blue to light blue
        return [f"rgb({50 + int(100 * i / max(n-1, 1))}, {50 + int(100 * i / max(n-1, 1))}, {180 + int(75 * i / max(n-1, 1))})" for i in range(n)]
    
    top_colors = generate_reds(n_top)
    bot_colors = generate_blues(n_bot)
    
    # Plot toplines (resistance)
    for i, scored in enumerate(scored_toplines):
        ruler = scored.ruler
        y_pred = ruler.predict(x)
        alpha = 0.9 - i * 0.05
        lw = 2.5 - i * 0.15
        
        color = top_colors[i] if i < len(top_colors) else "red"
        
        if i == 0:
            fig.line(x=x, y=y_pred, color=color, line_dash="dashed", 
                     line_width=lw, alpha=alpha, legend_label=f"Best Resistance: {scored.score}")
        else:
            fig.line(x=x, y=y_pred, color=color, line_dash="dashed", 
                     line_width=lw, alpha=alpha)
        
        # Annotate score
        score_label = Label(
            x=x[-1], y=y_pred[-1],
            text=f" {scored.score}",
            text_font_size="9pt", text_color=color, text_font_style="bold",
            x_offset=3, y_offset=0
        )
        fig.add_layout(score_label)
    
    # Plot bottomlines (support)
    for i, scored in enumerate(scored_bottomlines):
        ruler = scored.ruler
        y_pred = ruler.predict(x)
        alpha = 0.9 - i * 0.05
        lw = 2.5 - i * 0.15
        
        color = bot_colors[i] if i < len(bot_colors) else "blue"
        
        if i == 0:
            fig.line(x=x, y=y_pred, color=color, line_dash="dashed",
                     line_width=lw, alpha=alpha, legend_label=f"Best Support: {scored.score}")
        else:
            fig.line(x=x, y=y_pred, color=color, line_dash="dashed",
                     line_width=lw, alpha=alpha)
        
        # Annotate score
        score_label = Label(
            x=x[-1], y=y_pred[-1],
            text=f" {scored.score}",
            text_font_size="9pt", text_color=color, text_font_style="bold",
            x_offset=3, y_offset=0
        )
        fig.add_layout(score_label)
    
    # Move legend outside to the right
    if fig.legend:
        fig.add_layout(fig.legend[0], 'right')
        fig.legend.click_policy = "hide"
    
    return fig

