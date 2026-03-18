# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of points."

import operator
import typing
from abc import ABC
from collections.abc import Callable
from numbers import Number
from typing import Any, Self

import torch
from bokeh.plotting import figure as Figure
from torch import Tensor

from brokrest import tds

from .rects import Box
from .topos import Shape, Topo

__all__ = ["Vector", "Point"]


type ElemWiseRhs = "Number | Tensor | Vector"


@tds.tensorclass
class Vector(Topo, ABC):

    x: Tensor
    "The x element."

    y: Tensor
    "The y element."

    def __add__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(self, other, operator.add)

    def __radd__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(other, self, operator.add)

    def __sub__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(self, other, operator.sub)

    def __rsub__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(other, self, operator.sub)

    def __mul__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(self, other, operator.mul)

    def __rmul__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(other, self, operator.mul)

    def __truediv__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(self, other, operator.truediv)

    def __rtruediv__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(other, self, operator.truediv)

    def __floordiv__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(self, other, operator.floordiv)

    def __rfloordiv__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(other, self, operator.floordiv)

    def __pow__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(self, other, operator.pow)

    def __rpow__(self, other: ElemWiseRhs) -> Self:
        return _element_wise(other, self, operator.pow)

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

    def unit(self) -> Self:
        return self / self.length

    @property
    def length(self) -> Tensor:
        return (self.x**2 + self.y**2) ** 0.5


def _element_wise(lhs: Any, rhs: Any, op: Callable[..., Tensor]):
    # Even though both are `Vector`,
    # this would be called as a method on `lhs`,
    # so the return type is based on `lhs`.
    if isinstance(lhs, Vector) and isinstance(rhs, Vector):

        return type(lhs)(x=op(lhs.x, rhs.x), y=op(lhs.y, rhs.y))

    # For `__op__`.
    elif isinstance(lhs, Vector):

        return type(lhs)(x=op(lhs.x, rhs), y=op(lhs.y, rhs))

    # For `__rop__`.
    elif isinstance(rhs, Vector):
        return type(rhs)(x=op(lhs, rhs.x), y=op(lhs, rhs.y))

    # This will never happen (if called as metohd), so it shall raise and error.
    else:
        raise TypeError(f"Unrecognized types {type(lhs)=}, {type(rhs)=}.")


@tds.tensorclass
class Point(Vector, Shape):
    "A collection of points."

    @typing.override
    def _outer(self) -> Box:
        return Box(x_0=self.x, x_1=self.x, y_0=self.y, y_1=self.y)

    @typing.override
    def _draw(self, figure: Figure, /) -> None:
        _ = figure.scatter(
            x=self.x.numpy(),
            y=self.y.numpy(),
        )
