# Copyright (c) The BrokRest Authors - All Rights Reserved

"Polygons that works closly with shapely."

import typing

import numpy as np
import shapely
import tensordict as td
from bokeh import plotting

from brokrest.tds import TensorClass, tensorclass

from .lines import Point
from .rects import Segment
from .topos import Topo

__all__ = ["Polygon"]


@tensorclass
class Polygon(Topo):
    vertices: Point

    @typing.override
    def _setup_batch_size(self) -> None:
        if self.vertices.ndim == 0:
            raise ValueError("A single vertex cannot make a polygon.")

        self.batch_size = self.vertices.shape[1:]

    def segments(self) -> Segment:
        starts = self.vertices
        ends = self.vertices.roll(1, 0)
        return Segment.from_start_end(starts, ends)

    @typing.override
    def _draw(self, figure: plotting.figure, /) -> None:
        _ = self.segments()._draw(figure)

    @classmethod
    def from_segments(cls, *segments: Segment) -> typing.Self:
        batch = _maybe_stack_input(*segments)
        points = batch.points()
        return cls.from_vertices(points)

    @classmethod
    def from_vertices(cls, *vertices: Point) -> typing.Self:
        batch = _maybe_stack_input(*vertices)
        return cls(batch)

    @classmethod
    def from_shapely_polygon(cls, pg: shapely.Polygon) -> typing.Self:
        coords = np.array(pg.exterior.coords)
        points = Point(*coords.T)
        return cls.from_vertices(points)


def _maybe_stack_input[T: TensorClass](*items: T) -> T:
    if len(items) == 1:
        return items[0]
    else:
        return td.stack(items)
