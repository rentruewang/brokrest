# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of points."

import typing
from abc import ABC
from typing import ClassVar

from bokeh.plotting import figure as Figure

from .rects import Box
from .topos import Shape, Topo

__all__ = ["Vector", "Point"]


class Vector(Topo, ABC):
    KEYS: ClassVar[tuple[str, ...]] = "x", "y"

    @property
    def x(self):
        return self["x"]

    @property
    def y(self):
        return self["y"]


class Point(Vector, Shape):
    "A collection of points."

    @typing.override
    def _outer(self) -> Box:
        return Box.init(x_0=self.x, x_1=self.x, y_0=self.y, y_1=self.y)

    @typing.override
    def _draw(self, figure: Figure, /) -> None:
        _ = figure.scatter(
            x=self.x.numpy(),
            y=self.y.numpy(),
        )
