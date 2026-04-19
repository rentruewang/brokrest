# Copyright (c) The BrokRest Authors - All Rights Reserved

"Polygons that works closly with shapely."

import typing

import tensordict as td
from bokeh import plotting

from brokrest.tds import tensorclass

from .lines import Point
from .rects import Segment
from .topos import Topo

__all__ = ["Polygon"]


@tensorclass
class Polygon(Topo):
    vertices: Point

    def __post_init__(self) -> None:
        super().__post_init__()

    @typing.override
    def _setup_batch_size(self) -> None:
        if self.vertices.ndim == 0:
            raise ValueError("A single vertex cannot make a polygon.")

        self.batch_size = self.vertices.shape[:-1]

    @typing.override
    def _draw(self, figure: plotting.figure, /) -> None:
        raise NotImplementedError

    @classmethod
    def from_segments(cls, *segments: Segment) -> typing.Self:
        starts = [seg.start for seg in segments]
        ends = [seg.end for seg in segments]

        assert len(starts) == len(ends)

        if not all((start == end).all() for start, end in zip(starts[1:], ends)):
            raise ValueError("Segments are not connected.")

        return cls.from_vertices(*starts, ends[-1])

    @classmethod
    def from_vertices(cls, *vertices: Point):
        if len(vertices) == 1:
            batch = vertices[0]
        else:
            batch = td.cat(vertices)
        return cls(batch)
