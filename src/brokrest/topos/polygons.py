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

    @property
    def segments(self) -> Segment:
        starts = self.vertices
        ends = self.vertices.roll(1, 0)
        return Segment.from_start_end(starts, ends)

    def merge_segments(self, tolerance: float = 0.05) -> Segment:
        """
        Either merge consecutive segments, or drop segments that cannot be merged.
        """

        if (segments := self.segments).ndim != 1:
            raise ValueError("Only 1d segments can be supported right now.")

        slopes = segments.slope
        shifted = slopes.roll(1, 0)

        merge = (shifted - slopes).abs() <= tolerance
        idx_of_first_0 = [i for i, x in enumerate(merge) if not x][0]

        # Move the breaking point to the start s.t. we don't need to handle breakage.
        segments = segments.roll(-idx_of_first_0)
        merge = merge.roll(-idx_of_first_0)

        results: list[Segment] = []
        to_merge: list[Segment] = []
        for segment, do_merge in zip(segments, merge):
            if do_merge:
                to_merge.append(segment)

            elif to_merge:
                results.append(
                    Segment(
                        x_0=to_merge[0].x_0,
                        y_0=to_merge[0].y_0,
                        x_1=to_merge[-1].x_1,
                        y_1=to_merge[-1].y_1,
                    )
                )
                to_merge.clear()

            else:
                results.append(segment)

        return td.stack(results)

    @typing.override
    def _draw(self, figure: plotting.figure, /) -> None:
        _ = self.segments._draw(figure)

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
