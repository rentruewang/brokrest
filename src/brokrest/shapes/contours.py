# Copyright (c) The BrokRest Authors - All Rights Reserved

from __future__ import annotations

import dataclasses as dcls
from typing import TYPE_CHECKING

import numpy as np
from shapely import LineString, MultiPoint

from .histories import PriceHistory

if TYPE_CHECKING:
    from typing import Self


@dcls.dataclass(frozen=True)
class BoundingRange:
    lower: LineString
    upper: LineString

    @classmethod
    def from_history(cls, history: PriceHistory):
        return cls.from_points(history.points())

    @classmethod
    def from_points(cls, points: MultiPoint) -> BoundingRange:
        return _contours(points)


def _contours(points: MultiPoint):
    hull = LineString(points.convex_hull)
    xs, _ = np.array(hull.coords).T
    assert xs.ndim == 2
    assert xs.shape[1] == 2
    diff_xs = xs - np.roll(xs, shift=-1)
    upper = xs[diff_xs >= 0]
    lower = xs[diff_xs < 0]
    return BoundingRange(upper=upper, lower=lower)
