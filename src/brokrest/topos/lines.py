# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import typing

import torch

from brokrest.plotting import Canvas, Displayable
from brokrest.tds import tensorclass

from .topos import Topo
from .vecs import Point

__all__ = ["Line"]


@tensorclass
class Line(Displayable, Topo):
    """
    A set of lines. Represented as `y = mx + b` (slope intercept form).
    """

    m: torch.Tensor
    b: torch.Tensor

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
    def draw(self, canvas: Canvas) -> None:
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
