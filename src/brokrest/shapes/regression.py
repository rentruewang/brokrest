# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Piecewise linear regression for trend line detection.

Implements segmented regression to identify trend changes in price data.
"""

from __future__ import annotations

import dataclasses as dcls
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "TrendSegment",
    "PiecewiseRegression",
    "detect_trends",
    "optimal_breakpoints",
]


@dcls.dataclass(frozen=True)
class TrendSegment:
    """A single trend segment with linear regression parameters."""

    start_idx: int
    end_idx: int
    slope: float
    intercept: float
    r_squared: float

    @property
    def is_uptrend(self) -> bool:
        return self.slope > 0

    @property
    def is_downtrend(self) -> bool:
        return self.slope < 0

    def predict(self, x: NDArray) -> NDArray:
        """Predict y values for given x positions."""
        return self.slope * x + self.intercept

    def predict_range(self) -> tuple[float, float]:
        """Return (start_y, end_y) for the segment."""
        return (
            self.slope * self.start_idx + self.intercept,
            self.slope * self.end_idx + self.intercept,
        )


@dcls.dataclass
class PiecewiseRegression:
    """
    Piecewise linear regression for trend detection.

    Splits data into segments and fits linear regression to each.
    """

    segments: Sequence[TrendSegment]
    breakpoints: NDArray  # indices where trend changes
    total_r_squared: float

    @classmethod
    def fit(
        cls,
        x: NDArray,
        y: NDArray,
        n_segments: int = 5,
        min_segment_size: int = 10,
    ) -> PiecewiseRegression:
        """
        Fit piecewise linear regression with fixed number of segments.

        Args:
            x: X values (e.g., timestamps or indices)
            y: Y values (e.g., prices)
            n_segments: Number of segments to fit
            min_segment_size: Minimum points per segment
        """
        n = len(x)
        if n < n_segments * min_segment_size:
            n_segments = max(1, n // min_segment_size)

        # Find optimal breakpoints
        breakpoints = optimal_breakpoints(x, y, n_segments, min_segment_size)

        # Fit segments
        segments = []
        all_indices = [0] + list(breakpoints) + [n]

        total_ss_res = 0.0
        total_ss_tot = 0.0
        y_mean = np.mean(y)

        for i in range(len(all_indices) - 1):
            start = all_indices[i]
            end = all_indices[i + 1]

            x_seg = x[start:end].reshape(-1, 1)
            y_seg = y[start:end]

            model = LinearRegression()
            model.fit(x_seg, y_seg)

            y_pred = model.predict(x_seg)
            ss_res = np.sum((y_seg - y_pred) ** 2)
            ss_tot = np.sum((y_seg - np.mean(y_seg)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

            total_ss_res += ss_res
            total_ss_tot += np.sum((y_seg - y_mean) ** 2)

            segment = TrendSegment(
                start_idx=start,
                end_idx=end - 1,
                slope=float(model.coef_[0]),
                intercept=float(model.intercept_),
                r_squared=r_squared,
            )
            segments.append(segment)

        total_r_squared = 1 - (total_ss_res / total_ss_tot) if total_ss_tot > 0 else 1.0

        return cls(
            segments=segments,
            breakpoints=np.array(breakpoints),
            total_r_squared=total_r_squared,
        )

    @classmethod
    def fit_auto(
        cls,
        x: NDArray,
        y: NDArray,
        max_segments: int = 10,
        min_segment_size: int = 10,
        improvement_threshold: float = 0.02,
    ) -> PiecewiseRegression:
        """
        Automatically determine optimal number of segments.

        Increases segments until R² improvement falls below threshold.
        """
        best_result = cls.fit(x, y, n_segments=1, min_segment_size=min_segment_size)

        for n_seg in range(2, max_segments + 1):
            if len(x) < n_seg * min_segment_size:
                break

            result = cls.fit(x, y, n_segments=n_seg, min_segment_size=min_segment_size)
            improvement = result.total_r_squared - best_result.total_r_squared

            if improvement < improvement_threshold:
                break

            best_result = result

        return best_result

    def predict(self, x: NDArray) -> NDArray:
        """Predict y values using piecewise model."""
        y_pred = np.zeros_like(x, dtype=float)

        for seg in self.segments:
            mask = (np.arange(len(x)) >= seg.start_idx) & (np.arange(len(x)) <= seg.end_idx)
            y_pred[mask] = seg.predict(x[mask])

        return y_pred

    def trend_summary(self) -> str:
        """Return human-readable trend summary."""
        lines = []
        for i, seg in enumerate(self.segments):
            direction = "↑" if seg.is_uptrend else "↓" if seg.is_downtrend else "→"
            lines.append(
                f"Segment {i + 1}: {direction} "
                f"[{seg.start_idx}:{seg.end_idx}] "
                f"slope={seg.slope:.4f} R²={seg.r_squared:.3f}"
            )
        lines.append(f"Total R²: {self.total_r_squared:.3f}")
        return "\n".join(lines)


def optimal_breakpoints(
    x: NDArray,
    y: NDArray,
    n_segments: int,
    min_segment_size: int = 10,
) -> list[int]:
    """
    Find breakpoints using optimized binary segmentation.

    Uses vectorized operations and sampling for speed.
    """
    n = len(x)
    n_breaks = n_segments - 1

    if n_breaks == 0:
        return []

    def segment_cost_fast(start: int, end: int) -> float:
        """Fast segment cost using numpy (no sklearn overhead)."""
        if end - start < 2:
            return 0.0
        x_seg = x[start:end]
        y_seg = y[start:end]
        # Direct least squares: y = mx + b
        n_pts = len(x_seg)
        sum_x = np.sum(x_seg)
        sum_y = np.sum(y_seg)
        sum_xy = np.sum(x_seg * y_seg)
        sum_xx = np.sum(x_seg * x_seg)
        
        denom = n_pts * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-10:
            return np.sum((y_seg - np.mean(y_seg)) ** 2)
        
        m = (n_pts * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n_pts
        y_pred = m * x_seg + b
        return float(np.sum((y_seg - y_pred) ** 2))

    def find_best_split(start: int, end: int, step: int = 1) -> tuple[int, float]:
        """Find the best split point, optionally with step for speed."""
        best_idx = -1
        best_gain = -np.inf

        current_cost = segment_cost_fast(start, end)

        # Try split points with step
        for i in range(start + min_segment_size, end - min_segment_size + 1, step):
            left_cost = segment_cost_fast(start, i)
            right_cost = segment_cost_fast(i, end)
            gain = current_cost - (left_cost + right_cost)

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        return best_idx, best_gain

    # Binary segmentation with adaptive step
    breakpoints: list[int] = []
    segments = [(0, n)]

    for _ in range(n_breaks):
        best_split_idx = -1
        best_split_gain = -np.inf
        best_segment_idx = -1

        for seg_idx, (start, end) in enumerate(segments):
            seg_len = end - start
            if seg_len < 2 * min_segment_size:
                continue

            # Use larger step for longer segments, then refine
            step = max(1, seg_len // 50)
            split_idx, gain = find_best_split(start, end, step)
            
            # Refine around best point if step > 1
            if step > 1 and split_idx != -1:
                refine_start = max(start + min_segment_size, split_idx - step)
                refine_end = min(end - min_segment_size, split_idx + step)
                for i in range(refine_start, refine_end + 1):
                    left_cost = segment_cost_fast(start, i)
                    right_cost = segment_cost_fast(i, end)
                    current_cost = segment_cost_fast(start, end)
                    g = current_cost - (left_cost + right_cost)
                    if g > gain:
                        gain = g
                        split_idx = i

            if gain > best_split_gain:
                best_split_gain = gain
                best_split_idx = split_idx
                best_segment_idx = seg_idx

        if best_split_idx == -1:
            break

        start, end = segments[best_segment_idx]
        segments[best_segment_idx] = (start, best_split_idx)
        segments.insert(best_segment_idx + 1, (best_split_idx, end))
        breakpoints.append(best_split_idx)

    return sorted(breakpoints)


def detect_trends(
    prices: NDArray,
    timestamps: NDArray | None = None,
    n_segments: int | None = None,
    auto: bool = True,
    **kwargs,
) -> PiecewiseRegression:
    """
    Convenience function to detect trends in price data.

    Args:
        prices: Price array
        timestamps: Optional timestamp array (uses indices if None)
        n_segments: Number of segments (ignored if auto=True)
        auto: Automatically determine optimal segments
        **kwargs: Additional arguments to PiecewiseRegression.fit

    Returns:
        PiecewiseRegression with detected trends
    """
    if timestamps is None:
        x = np.arange(len(prices), dtype=float)
    else:
        x = timestamps.astype(float)

    y = np.asarray(prices, dtype=float)

    if auto:
        return PiecewiseRegression.fit_auto(x, y, **kwargs)
    else:
        return PiecewiseRegression.fit(x, y, n_segments=n_segments or 5, **kwargs)

