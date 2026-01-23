# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of points."

import typing
from typing import ClassVar

from bokeh.plotting import figure as Figure

from .rects import Box
from .topos import Shape

__all__ = ["Point"]


class Point(Shape):
    "A collection of points."

    KEYS: ClassVar[tuple[str, ...]] = "x", "y"

    @property
    def x(self):
        return self["x"]

    @property
    def y(self):
        return self["y"]

    @typing.override
    def _outer(self) -> Box:
        return Box.init_tensor(x_0=self.x, x_1=self.x, y_0=self.y, y_1=self.y)

    @typing.override
    def _draw(self, figure: Figure, /) -> None:
        _ = figure.scatter(
            x=self.x.numpy(),
            y=self.y.numpy(),
        )
