# Copyright (c) The BrokRest Authors - All Rights Reserved

"""
Convex hull contour calculation for price data.
"""

from __future__ import annotations

import dataclasses as dcls

import numpy as np
from numpy.typing import NDArray
from shapely import MultiPoint
from shapely.geometry import LineString

__all__ = ["BoundingRange", "compute_contours"]


@dcls.dataclass(frozen=True)
class BoundingRange:
    """Upper and lower bounding lines from convex hull."""

    upper: NDArray
    lower: NDArray

    @classmethod
    def from_prices(cls, prices: NDArray) -> BoundingRange:
        """Create bounding range from price array."""
        x = np.arange(len(prices))
        points = MultiPoint(list(zip(x, prices)))
        return compute_contours(points)


def compute_contours(points: MultiPoint) -> BoundingRange:
    """
    Compute upper and lower contours from convex hull.
    
    Args:
        points: MultiPoint geometry
        
    Returns:
        BoundingRange with upper and lower bounds
    """
    hull = points.convex_hull
    
    if hull.is_empty or hull.geom_type == "Point":
        return BoundingRange(upper=np.array([]), lower=np.array([]))
    
    if hull.geom_type == "LineString":
        coords = np.array(hull.coords)
        return BoundingRange(upper=coords, lower=coords)
    
    coords = np.array(hull.exterior.coords)
    xs = coords[:, 0]
    
    # Find leftmost and rightmost points
    left_idx = np.argmin(xs)
    right_idx = np.argmax(xs)
    
    n = len(coords)
    
    # Upper hull: from left to right (counter-clockwise)
    if left_idx <= right_idx:
        upper_indices = list(range(left_idx, right_idx + 1))
    else:
        upper_indices = list(range(left_idx, n)) + list(range(0, right_idx + 1))
    
    # Lower hull: from right to left
    if right_idx <= left_idx:
        lower_indices = list(range(right_idx, left_idx + 1))
    else:
        lower_indices = list(range(right_idx, n)) + list(range(0, left_idx + 1))
    
    return BoundingRange(
        upper=coords[upper_indices],
        lower=coords[lower_indices],
    )
