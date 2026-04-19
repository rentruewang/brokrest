# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import typing

import torch
from bokeh import plotting

from brokrest.tds import tensorclass

from .rects import Box
from .topos import Topo

__all__ = ["Line", "Point"]


@tensorclass
class Point(Topo):
    "A collection of points."

    x: torch.Tensor
    "The x element."

    y: torch.Tensor
    "The y element."

    @typing.override
    def _outer(self) -> Box:
        return Box(x_0=self.x, x_1=self.x, y_0=self.y, y_1=self.y)

    def unit(self) -> typing.Self:
        return self / self.length

    @property
    def length(self) -> float:
        return (self.x**2 + self.y**2).sum().item() ** 0.5

    @typing.override
    def _draw(self, figure: plotting.figure, /) -> None:
        _ = figure.scatter(
            x=self.x.numpy(),
            y=self.y.numpy(),
        )


@tensorclass
class Line(Topo):
    """
    A set of lines. Represented as `y = mx + b` (slope intercept form).
    """

    m: torch.Tensor
    "The slope of the line."

    b: torch.Tensor
    "The bias of the line."

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns mx + b as a self.ndim + 1 matrix `R`. `R_ij = m_i x_j + b_i.`
        """

        return self.m[..., None] * x[None, ...] + self.b[..., None]

    def subs(self, points: Point) -> torch.Tensor:
        """
        Returns mx + b as a self.ndim + 1 matrix `R`.
        `R_ij = m_i x_j + b_i - y_j.`
        """

        return self.apply(points.x) - points.y[None, ...]

    @typing.override
    def _outer(self) -> "Box":
        return NotImplemented

    @typing.override
    def _draw(self, canvas: plotting.figure) -> None:
        raise NotImplementedError

    @classmethod
    def standard(cls, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> typing.Self:
        "Create a line in the `ax + by + c = 0` form."

        # y = -a/b x - c/b
        return cls(m=-a / b, b=-c / b)

    @classmethod
    def intercept(cls, a: torch.Tensor, b: torch.Tensor) -> typing.Self:
        "Create a line in the `x/a + y/b = 1` form."

        # y = b - b/a x
        return cls(m=-b / a, b=b)

    @classmethod
    def slope_intercept(cls, m: torch.Tensor, b: torch.Tensor) -> typing.Self:
        "Create a line in the `y = mx + b` form."

        return cls(m=m, b=b)
