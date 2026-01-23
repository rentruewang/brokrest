# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import typing
from typing import ClassVar, Self

from torch import Tensor

from brokrest.plotting import Canvas

from .points import Point
from .topos import Topo

__all__ = ["Line"]


class Line(Topo):
    """
    A set of lines. Represented as `y = mx + b` (slope intercept form).
    """

    KEYS: ClassVar[tuple[str, ...]] = "m", "b"

    def apply(self, x: Tensor) -> Tensor:
        """
        Returns mx + b as a self.ndim + 1 matrix ``R``. ``R_ij = m_i x_j + b_i.``
        """

        return self.m[..., None] * x[None, ...] + self.b[..., None]

    def subs(self, points: Point) -> Tensor:
        """
        Returns mx + b as a self.ndim + 1 matrix ``R``.
        ``R_ij = m_i x_j + b_i - y_j.``
        """

        return self.apply(points.x) - points.y[None, ...]

    @property
    def m(self):
        return self["m"]

    @property
    def b(self):
        return self["b"]

    @typing.override
    def draw(self, canvas: Canvas) -> None:
        raise NotImplementedError

    @classmethod
    def standard(cls, a: Tensor, b: Tensor, c: Tensor) -> Self:
        "Create a line in the ``ax + by + c = 0`` form."

        # y = -a/b x - c/b
        return cls.init_tensor(m=-a / b, b=-c / b)

    @classmethod
    def intercept(cls, a: Tensor, b: Tensor) -> Self:
        "Create a line in the ``x/a + y/b = 1`` form."

        # y = b - b/a x
        return cls.init_tensor(m=-b / a, b=b)

    @classmethod
    def slope_intercept(cls, m: Tensor, b: Tensor) -> Self:
        "Create a line in the ``y = mx + b`` form."

        return cls.init_tensor(m=m, b=b)
