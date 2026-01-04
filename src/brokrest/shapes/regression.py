# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Piecewise linear regression for trend line detection.

Implements segmented regression to identify trend changes in price data.
Uses binary segmentation with O(1) RSS computation for efficiency.
"""

from __future__ import annotations

import dataclasses as dcls
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from brokrest.equations import SlopeInterceptForm
from brokrest.painters import Canvas, Painter
from brokrest.vectors import Vec2d

__all__ = [
    "TrendSegment",
    "PiecewiseRegression",
    "detect_trends",
]


@dcls.dataclass(frozen=True)
class TrendSegment(Painter):
    """
    A single trend segment with linear regression parameters.
    
    Implements Painter protocol for matplotlib integration.
    """

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

    @property
    def equation(self) -> SlopeInterceptForm:
        """Linear equation: y = mx + b"""
        return SlopeInterceptForm(m=self.slope, b=self.intercept)

    def predict(self, x: NDArray) -> NDArray:
        """Predict y values for given x positions."""
        return self.slope * x + self.intercept

    def predict_range(self) -> tuple[float, float]:
        """Return (start_y, end_y) for the segment."""
        return (
            self.slope * self.start_idx + self.intercept,
            self.slope * self.end_idx + self.intercept,
        )

    def plot(self, canvas: Canvas) -> None:
        """Draw trend line on matplotlib canvas."""
        y_start, y_end = self.predict_range()
        color = "green" if self.is_uptrend else "red"
        canvas.line(
            Vec2d(self.start_idx, y_start),
            Vec2d(self.end_idx, y_end),
            color=color,
        )


@dcls.dataclass
class PiecewiseRegression(Painter):
    """
    Piecewise linear regression for trend detection.

    Splits data into segments and fits linear regression to each.
    """

    segments: Sequence[TrendSegment]
    breakpoints: NDArray
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
            x: X values (indices)
            y: Y values (prices)
            n_segments: Number of segments to fit
            min_segment_size: Minimum points per segment
        """
        n = len(x)
        if n < n_segments * min_segment_size:
            n_segments = max(1, n // min_segment_size)

        breakpoints = _optimal_breakpoints(x, y, n_segments, min_segment_size)

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
        best = cls.fit(x, y, n_segments=1, min_segment_size=min_segment_size)

        for n_seg in range(2, max_segments + 1):
            if len(x) < n_seg * min_segment_size:
                break

            result = cls.fit(x, y, n_segments=n_seg, min_segment_size=min_segment_size)
            improvement = result.total_r_squared - best.total_r_squared

            if improvement < improvement_threshold:
                break

            best = result

        return best

    def predict(self, x: NDArray) -> NDArray:
        """Predict y values using piecewise model."""
        y_pred = np.zeros_like(x, dtype=float)

        for seg in self.segments:
            mask = (np.arange(len(x)) >= seg.start_idx) & (np.arange(len(x)) <= seg.end_idx)
            y_pred[mask] = seg.predict(x[mask])

        return y_pred

    def plot(self, canvas: Canvas) -> None:
        """Draw all trend segments on canvas."""
        for seg in self.segments:
            seg.plot(canvas)

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


def _optimal_breakpoints(
    x: NDArray,
    y: NDArray,
    n_segments: int,
    min_segment_size: int = 10,
) -> list[int]:
    """
    Find breakpoints using greedy binary segmentation.

    Uses O(1) RSS computation via prefix sums for efficiency.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    n_breaks = n_segments - 1
    if n_breaks <= 0:
        return []

    if n < (n_segments * min_segment_size):
        n_breaks = max(0, (n // min_segment_size) - 1)
        if n_breaks == 0:
            return []

    # Prefix sums for O(1) segment stats
    sx = np.zeros(n + 1, dtype=float)
    sy = np.zeros(n + 1, dtype=float)
    sxx = np.zeros(n + 1, dtype=float)
    sxy = np.zeros(n + 1, dtype=float)
    syy = np.zeros(n + 1, dtype=float)

    sx[1:] = np.cumsum(x)
    sy[1:] = np.cumsum(y)
    sxx[1:] = np.cumsum(x * x)
    sxy[1:] = np.cumsum(x * y)
    syy[1:] = np.cumsum(y * y)

    eps = 1e-12

    def segment_rss(start: int, end: int) -> float:
        """RSS of best-fit line over [start,end), computed in O(1)."""
        if end - start < 2:
            return 0.0

        n_pts = end - start
        sum_x = sx[end] - sx[start]
        sum_y = sy[end] - sy[start]
        sum_xx = sxx[end] - sxx[start]
        sum_xy = sxy[end] - sxy[start]
        sum_yy = syy[end] - syy[start]

        if n_pts <= 1:
            return 0.0

        denom = n_pts * sum_xx - (sum_x * sum_x)
        if abs(denom) < eps:
            rss = sum_yy - (sum_y * sum_y) / max(n_pts, 1)
            return float(max(0.0, rss))

        m = (n_pts * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n_pts

        rss = (
            sum_yy
            + (m * m) * sum_xx
            + 2.0 * m * b * sum_x
            + (b * b) * n_pts
            - 2.0 * m * sum_xy
            - 2.0 * b * sum_y
        )

        if rss < 0 and rss > -1e-9:
            rss = 0.0
        return float(rss)

    def find_best_split(start: int, end: int) -> tuple[int, float]:
        """Exact best split for [start,end) subject to min_segment_size."""
        current = segment_rss(start, end)

        best_idx = -1
        best_gain = -np.inf

        lo = start + min_segment_size
        hi = end - min_segment_size
        if lo > hi:
            return -1, -np.inf

        for i in range(lo, hi + 1):
            left = segment_rss(start, i)
            right = segment_rss(i, end)
            gain = current - (left + right)
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        return best_idx, float(best_gain)

    breakpoints: list[int] = []
    segments: list[tuple[int, int]] = [(0, n)]

    for _ in range(n_breaks):
        best_split_idx = -1
        best_split_gain = -np.inf
        best_segment_idx = -1

        for seg_idx, (start, end) in enumerate(segments):
            if end - start < 2 * min_segment_size:
                continue
            split_idx, gain = find_best_split(start, end)
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
    Detect trends in price data.

    Args:
        prices: Price array
        timestamps: Optional timestamp array (uses indices if None)
        n_segments: Number of segments (ignored if auto=True)
        auto: Automatically determine optimal segments
        **kwargs: Additional arguments to fit

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
