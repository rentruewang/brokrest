# Copyright (c) The BrokRest Authors - All Rights Reserved

"Polygons that works closly with shapely."

import shapely, typing
from .topos import Topo
from .lines import Line
from .rects import Segment
from collections.abc import Sequence
from .lines import Point
from bokeh import plotting
from brokrest.tds import tensorclass


@tensorclass
class Polygon(Topo):
    points: Point

    @typing.override
    def _draw(self, figure: plotting.figure, /) -> None:
        raise NotImplementedError

    @classmethod
    def from_segments(cls, *segments: Segment) -> typing.Self:
        return cls()
