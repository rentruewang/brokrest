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
    "detect_spikes",
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
    Find breakpoints using greedy binary segmentation.

    Improvements vs current implementation:
    - O(1) RSS evaluation using prefix sums (no slicing sums)
    - Exact best split per segment (step=1) is now fast and avoids approximation errors
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    n_breaks = n_segments - 1
    if n_breaks <= 0:
        return []

    if n < (n_segments * min_segment_size):
        # Caller adjusts n_segments already, but keep this safe
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

    def _seg_sums(start: int, end: int) -> tuple[int, float, float, float, float, float]:
        """Return n, sum_x, sum_y, sum_xx, sum_xy, sum_yy for [start,end)."""
        n_pts = end - start
        sum_x = sx[end] - sx[start]
        sum_y = sy[end] - sy[start]
        sum_xx = sxx[end] - sxx[start]
        sum_xy = sxy[end] - sxy[start]
        sum_yy = syy[end] - syy[start]
        return n_pts, sum_x, sum_y, sum_xx, sum_xy, sum_yy

    def segment_rss(start: int, end: int) -> float:
        """RSS of best-fit line y = m x + b over [start,end), computed in O(1)."""
        if end - start < 2:
            return 0.0

        n_pts, sum_x, sum_y, sum_xx, sum_xy, sum_yy = _seg_sums(start, end)
        if n_pts <= 1:
            return 0.0

        denom = n_pts * sum_xx - (sum_x * sum_x)
        if abs(denom) < eps:
            # Fall back to constant model: RSS = sum((y - mean)^2)
            rss = sum_yy - (sum_y * sum_y) / max(n_pts, 1)
            return float(max(0.0, rss))

        m = (n_pts * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n_pts

        # RSS = Σ(y - (mx+b))^2 expanded using sums:
        rss = (
            sum_yy
            + (m * m) * sum_xx
            + 2.0 * m * b * sum_x
            + (b * b) * n_pts
            - 2.0 * m * sum_xy
            - 2.0 * b * sum_y
        )

        # Numerical clamp
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
def detect_spikes(
    y: NDArray,
    window: int = 5,
    prominence: float = 0.0,
    min_distance: int = 10,
) -> tuple[list[int], list[int]]:
    """
    Detect local maxima (peaks) and minima (valleys) in price data.

    Args:
        y: Price array
        window: Window size for local extrema detection
        prominence: Minimum prominence (relative height) for a spike
                   0.0 = any local extrema, higher = more significant only
        min_distance: Minimum distance between detected spikes

    Returns:
        (peak_indices, valley_indices)
    """
    n = len(y)
    if n < 2 * window + 1:
        return [], []

    peaks = []
    valleys = []

    # Calculate prominence threshold based on price range
    price_range = np.max(y) - np.min(y)
    prominence_threshold = prominence * price_range

    for i in range(window, n - window):
        left_window = y[i - window:i]
        right_window = y[i + 1:i + window + 1]

        # Check for local maximum
        if y[i] > np.max(left_window) and y[i] > np.max(right_window):
            # Calculate prominence (height above surrounding valleys)
            left_min = np.min(y[max(0, i - window * 2):i])
            right_min = np.min(y[i + 1:min(n, i + window * 2 + 1)])
            prom = y[i] - max(left_min, right_min)

            if prom >= prominence_threshold:
                peaks.append(i)

        # Check for local minimum
        elif y[i] < np.min(left_window) and y[i] < np.min(right_window):
            # Calculate prominence (depth below surrounding peaks)
            left_max = np.max(y[max(0, i - window * 2):i])
            right_max = np.max(y[i + 1:min(n, i + window * 2 + 1)])
            prom = min(left_max, right_max) - y[i]

            if prom >= prominence_threshold:
                valleys.append(i)

    # Filter by minimum distance
    def filter_by_distance(indices: list[int], values: NDArray, is_peak: bool) -> list[int]:
        if not indices:
            return []

        filtered = [indices[0]]
        for idx in indices[1:]:
            if idx - filtered[-1] >= min_distance:
                filtered.append(idx)
            else:
                # Keep the more extreme one
                if is_peak:
                    if values[idx] > values[filtered[-1]]:
                        filtered[-1] = idx
                else:
                    if values[idx] < values[filtered[-1]]:
                        filtered[-1] = idx
        return filtered

    peaks = filter_by_distance(peaks, y, is_peak=True)
    valleys = filter_by_distance(valleys, y, is_peak=False)

    return peaks, valleys


def spike_aware_breakpoints(
    x: NDArray,
    y: NDArray,
    n_segments: int,
    min_segment_size: int = 10,
    spike_window: int = 5,
    spike_prominence: float = 0.05,
) -> list[int]:
    """
    Find breakpoints that respect local spike points.

    Strategy:
    1. Detect all significant spikes (local min/max)
    2. Use spikes as candidate breakpoints
    3. Fill in additional breakpoints between spikes if needed

    Args:
        x, y: Data arrays
        n_segments: Target number of segments
        min_segment_size: Minimum points per segment
        spike_window: Window for spike detection
        spike_prominence: Minimum prominence (0-1 scale of price range)

    Returns:
        List of breakpoint indices
    """
    n = len(x)
    n_breaks = n_segments - 1

    if n_breaks == 0:
        return []

    # Step 1: Detect spikes
    peaks, valleys = detect_spikes(
        y,
        window=spike_window,
        prominence=spike_prominence,
        min_distance=min_segment_size,
    )

    # Combine and sort all spike indices
    all_spikes = sorted(set(peaks + valleys))

    # Filter out spikes too close to edges
    all_spikes = [s for s in all_spikes if min_segment_size <= s <= n - min_segment_size]

    # Step 2: Select breakpoints from spikes
    if len(all_spikes) >= n_breaks:
        # More spikes than needed: select the most prominent ones
        spike_scores = []
        for idx in all_spikes:
            # Score based on local prominence
            left_range = y[max(0, idx - spike_window * 2):idx]
            right_range = y[idx + 1:min(n, idx + spike_window * 2 + 1)]

            if len(left_range) > 0 and len(right_range) > 0:
                if idx in peaks:
                    score = y[idx] - (np.min(left_range) + np.min(right_range)) / 2
                else:
                    score = (np.max(left_range) + np.max(right_range)) / 2 - y[idx]
            else:
                score = 0

            spike_scores.append((idx, score))

        # Sort by score and take top n_breaks
        spike_scores.sort(key=lambda x: -x[1])
        breakpoints = sorted([s[0] for s in spike_scores[:n_breaks]])

    elif len(all_spikes) > 0:
        # Fewer spikes than needed: use all spikes and fill with regular method
        breakpoints = list(all_spikes)

        # Fill remaining breakpoints using regular method
        remaining = n_breaks - len(breakpoints)
        if remaining > 0:
            # Find gaps between existing breakpoints
            all_points = [0] + sorted(breakpoints) + [n]
            gaps = []
            for i in range(len(all_points) - 1):
                gap_start = all_points[i]
                gap_end = all_points[i + 1]
                gap_size = gap_end - gap_start
                if gap_size >= 2 * min_segment_size:
                    gaps.append((gap_start, gap_end, gap_size))

            # Sort gaps by size, add breakpoints to largest gaps
            gaps.sort(key=lambda x: -x[2])

            for gap_start, gap_end, _ in gaps[:remaining]:
                # Find best split in this gap using cost function
                best_idx = (gap_start + gap_end) // 2  # Default to middle

                # Simple cost-based refinement
                best_cost = float('inf')
                for i in range(gap_start + min_segment_size, gap_end - min_segment_size + 1):
                    left_y = y[gap_start:i]
                    right_y = y[i:gap_end]

                    if len(left_y) < 2 or len(right_y) < 2:
                        continue

                    # Simple variance-based cost
                    cost = np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = i

                breakpoints.append(best_idx)

        breakpoints = sorted(breakpoints)[:n_breaks]

    else:
        # No spikes detected: fall back to regular method
        breakpoints = optimal_breakpoints(x, y, n_segments, min_segment_size)

    return sorted(breakpoints)


def detect_trends(
    prices: NDArray,
    timestamps: NDArray | None = None,
    n_segments: int | None = None,
    auto: bool = True,
    spike_mode: bool = False,
    spike_window: int = 5,
    spike_prominence: float = 0.05,
    **kwargs,
) -> PiecewiseRegression:
    """
    Convenience function to detect trends in price data.

    Args:
        prices: Price array
        timestamps: Optional timestamp array (uses indices if None)
        n_segments: Number of segments (ignored if auto=True)
        auto: Automatically determine optimal segments
        spike_mode: Use spike-aware breakpoint detection
        spike_window: Window size for spike detection (spike_mode only)
        spike_prominence: Min prominence 0-1 for spikes (spike_mode only)
        **kwargs: Additional arguments to PiecewiseRegression.fit

    Returns:
        PiecewiseRegression with detected trends
    """
    if timestamps is None:
        x = np.arange(len(prices), dtype=float)
    else:
        x = timestamps.astype(float)

    y = np.asarray(prices, dtype=float)

    if spike_mode:
        # Use spike-aware method
        if auto:
            # Auto-determine segment count, then use spike-aware breakpoints
            # First, get segment count from regular auto
            temp_result = PiecewiseRegression.fit_auto(x, y, **kwargs)
            n_seg = len(temp_result.segments)
        else:
            n_seg = n_segments or 5

        min_seg_size = kwargs.get('min_segment_size', 10)
        breakpoints = spike_aware_breakpoints(
            x, y, n_seg, min_seg_size,
            spike_window=spike_window,
            spike_prominence=spike_prominence,
        )

        # Fit segments using these breakpoints
        return _fit_with_breakpoints(x, y, breakpoints)
    else:
        # Original method
        if auto:
            return PiecewiseRegression.fit_auto(x, y, **kwargs)
        else:
            return PiecewiseRegression.fit(x, y, n_segments=n_segments or 5, **kwargs)


def _fit_with_breakpoints(
    x: NDArray,
    y: NDArray,
    breakpoints: list[int],
) -> PiecewiseRegression:
    """Fit piecewise regression with pre-determined breakpoints."""
    n = len(x)
    segments = []
    all_indices = [0] + list(breakpoints) + [n]

    total_ss_res = 0.0
    total_ss_tot = 0.0
    y_mean = np.mean(y)

    for i in range(len(all_indices) - 1):
        start = all_indices[i]
        end = all_indices[i + 1]

        if end <= start:
            continue

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

    return PiecewiseRegression(
        segments=segments,
        breakpoints=np.array(breakpoints),
        total_r_squared=total_r_squared,
    )

