# Copyright (c) The BrokRest Authors - All Rights Reserved

"Polygons that works closly with shapely."

import dataclasses as dcls
import typing

import numpy as np
import shapely
import tensordict as td
from bokeh import plotting

from .lines import Line, Point
from .rects import Segment
from .topos import Topo

__all__ = ["Polygon"]


@dcls.dataclass
class Polygon(Topo):
    upper: Point
    lower: Point
    left: Point
    right: Point

    @typing.override
    def _setup_shape(self):
        if self.vertices.ndim == 0:
            raise ValueError("A single vertex cannot make a polygon.")

        return self.vertices.shape[:-1]

    @property
    def vertices(self) -> Point:
        return td.cat(
            [self.upper, self.lower, td.stack([self.left, self.right], dim=-1)], dim=-1
        )

    @property
    def segments(self) -> Segment:
        ub = Segment.from_points(self.upper_bound).face_right()
        lb = Segment.from_points(self.lower_bound).face_right()
        return td.cat([ub, lb])

    @property
    def upper_bound(self) -> Point:
        return td.cat([self.left[None], self.upper, self.right[None]])

    @property
    def lower_bound(self) -> Point:
        return td.cat([self.left[None], self.lower, self.right[None]])

    @typing.override
    def plot(self, figure: plotting.figure, /) -> None:
        _ = self.segments.plot(figure)

    @classmethod
    def from_vertices(cls, *vertices: Point) -> typing.Self:
        batch = _maybe_stack_input(*vertices)

        point_list = td.stack(sorted(batch, key=lambda p: p.x))
        left, right = point_list[0], point_list[-1]

        line = Line.from_segment(Segment.from_start_end(left, right))

        val = line.subs(point_list).flatten()

        lr = np.isclose(val, 0, atol=1e-4)
        assert lr.sum() == 2

        upper: Point = point_list[val > 0 & ~lr]
        lower: Point = point_list[val < 0 & ~lr]

        return cls(left=left, right=right, upper=upper, lower=lower)

    @classmethod
    def from_shapely_polygon(cls, pg: shapely.Polygon) -> typing.Self:
        coords = np.array(pg.exterior.coords)
        points = Point(*coords.T)
        return cls.from_vertices(points)


def _maybe_stack_input[T](*items: T) -> T:
    if len(items) == 1:
        return items[0]
    else:
        return td.stack(items)
