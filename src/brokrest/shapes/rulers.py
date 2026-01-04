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

__all__ = ["Ruler", "find_rulers"]


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


def _optimize_topline_with_band(
    x: NDArray,
    y: NDArray,
    m: float,
    b: float,
    tolerance_factor: float = 0.1,
    invalid_penalty: float = 0.0,
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
        
    Returns:
        (optimized_slope, optimized_intercept)
    """
    band = tolerance_factor * np.std(y)
    if band < 1e-10:
        return m, b
    
    # Current line prediction
    pred = m * x + b
    
    # For topline: distance = pred - y (positive means point is below line)
    distances = pred - y
    
    # Current: points in band are those with |distance| <= band
    in_band_current = np.sum(np.abs(distances) <= band)
    # Invalid points: above the line (distance < 0)
    invalid_current = np.sum(distances < 0)
    current_score = in_band_current - invalid_penalty * invalid_current
    
    # If already have many points in band, no need to optimize
    if in_band_current >= len(x) * 0.3:
        return m, b
    
    # Try shifting the line down to where more points cluster
    # Key idea: find the shift that maximizes (in_band - invalid_penalty * invalid)
    
    best_b = b
    best_score = current_score
    
    # Try different shift amounts: target each point to be AT the line
    for target_idx in range(len(x)):
        # Shift so this point is exactly on the line
        # new line: y = m*x + new_b, we want m*x[target_idx] + new_b = y[target_idx]
        new_b = y[target_idx] - m * x[target_idx]
        
        new_pred = m * x + new_b
        new_distances = new_pred - y
        
        # Count points in band (within band distance of line)
        in_band = np.sum(np.abs(new_distances) <= band)
        
        # Invalid points: above the line (distance < 0, i.e., y > pred)
        invalid = np.sum(new_distances < 0)
        
        # Score: maximize in_band - invalid_penalty * invalid
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
            x, y, m_opt, b_opt, tolerance_factor, invalid_penalty
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
            x, y, m_opt, b_opt, tolerance_factor, invalid_penalty
        )
    
    return m_opt, b_opt


def _optimize_bottomline_with_band(
    x: NDArray,
    y: NDArray,
    m: float,
    b: float,
    tolerance_factor: float = 0.1,
    invalid_penalty: float = 0.0,
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
    """
    band = tolerance_factor * np.std(y)
    if band < 1e-10:
        return m, b
    
    # Current line prediction
    pred = m * x + b
    
    # For bottomline: distance = y - pred (positive means point is above line)
    distances = y - pred
    
    # Current: points in band are those with |distance| <= band
    in_band_current = np.sum(np.abs(distances) <= band)
    # Invalid points: below the line (distance < 0)
    invalid_current = np.sum(distances < 0)
    current_score = in_band_current - invalid_penalty * invalid_current
    
    # If already have many points in band, no need to optimize
    if in_band_current >= len(x) * 0.3:
        return m, b
    
    best_b = b
    best_score = current_score
    
    # Try different shift amounts: target each point to be AT the line
    for target_idx in range(len(x)):
        new_b = y[target_idx] - m * x[target_idx]
        
        new_pred = m * x + new_b
        new_distances = y - new_pred
        
        # Count points in band
        in_band = np.sum(np.abs(new_distances) <= band)
        
        # Invalid points: below the line (distance < 0, i.e., y < pred)
        invalid = np.sum(new_distances < 0)
        
        # Score: maximize in_band - invalid_penalty * invalid
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
            clamp=clamp, invalid_penalty=invalid_penalty
        )
        # Find bottomline (support) with slope optimization
        m_bot, b_bot = _find_bottomline(
            x, y, tolerance=tolerance, tolerance_factor=tolerance_factor, 
            clamp=clamp, invalid_penalty=invalid_penalty
        )
    else:
        # Parallel lines: only shift, no rotation
        m_top, b_top = _find_topline_parallel(x, y)
        m_bot, b_bot = _find_bottomline_parallel(x, y)
    
    topline = Ruler(slope=m_top, intercept=b_top, is_top=True)
    bottomline = Ruler(slope=m_bot, intercept=b_bot, is_top=False)
    
    return topline, bottomline


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
    
    # Plot topline (resistance)
    ax.plot(x, y_top, color="red", linestyle="--", linewidth=2,
            label=f"Resistance (y={topline.slope:.4f}x+{topline.intercept:.2f})")
    
    # Plot bottomline (support)
    ax.plot(x, y_bot, color="blue", linestyle="--", linewidth=2,
            label=f"Support (y={bottomline.slope:.4f}x+{bottomline.intercept:.2f})")
    
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
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
    
    # Plot topline (resistance)
    fig.line(x=x, y=y_top, color="red", line_dash="dashed", line_width=2,
             legend_label="Resistance")
    
    # Plot bottomline (support)
    fig.line(x=x, y=y_bot, color="blue", line_dash="dashed", line_width=2,
             legend_label="Support")
    
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    
    return fig

