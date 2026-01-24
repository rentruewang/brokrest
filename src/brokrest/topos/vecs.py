# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of points."

import typing
from abc import ABC
from typing import ClassVar

import torch
from bokeh.plotting import figure as Figure
from torch import Tensor

from .rects import Box
from .topos import Shape, Topo

__all__ = ["Vector", "Point"]


class Vector(Topo, ABC):
    KEYS: ClassVar[tuple[str, ...]] = "x", "y"

    def __matmul__(self, other: "Vector") -> Tensor:
        """
        Creates a m x n matrix O if len(self) = m and len(other) = n,
        where O[p, q] = self[p] * other[q].

        Args:
            other: Another vector.

        Returns:
            A 2D matrix.
        """

        outer = lambda l, r, /: torch.einsum("m,n->mn", l, r)
        return outer(self.x, other.x) + outer(self.y, other.y)

    @property
    def x(self):
        return self["x"]

    @property
    def y(self):
        return self["y"]

    def unit(self):
        return Vector(data=self.data / self.length)

    @property
    def length(self) -> Tensor:
        return (self.x**2 + self.y**2) ** 0.5


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
