# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Sequence
from typing import TypeAlias, TypeIs

import torch
from torch import Tensor

from brokrest.vectors import Vec2d

from .painters import Canvas

__all__ = ["LinearEq", "StandardForm", "SlopeInterceptForm", "InterceptForm"]

_TensorLike: TypeAlias = int | float | Tensor


class LinearEq(ABC):
    """
    A linear equation.
    """

    def solve(self, x: _TensorLike, /) -> Tensor:
        """
        Get the y value of the line equation when x is given.

        Args:
            x: The x value (can be a tensor).

        Returns:
            The y value (a tensor).
        """

        x = _promote_to_tensor(x)
        return self._solve(x)

    def subs(self, x: _TensorLike, y: _TensorLike) -> Tensor:
        """
        The linear equation assumes that RHS = 0.
        This method applies x, y to the LHS, and get the value of RHS.

        If this method returns 0, the point is on the line.
        If 2 points (a, b), (c, d) have the same signs (both positive or both negative)
        after substituted into the equation, they are on the same side of the line.

        Args:
            x: The x values (can be a tensor, list, or scalar).
            y: The y values (can be a tensor, list, or scalar).

        Returns:
            The result tensor.
        """

        x = _promote_to_tensor(x)
        y = _promote_to_tensor(y)

        # To make the implementation easier below.
        x, y = torch.broadcast_tensors(x, y)
        assert x.ndim in [0, 1], f"Should have been a 1D or 2D tensor. Got {x.ndim}D."
        assert x.shape == y.shape

        return self._subs(x, y)

    @abc.abstractmethod
    def _solve(self, x: Tensor) -> Tensor:
        "The implementation for ``.solve``."

        ...

    @abc.abstractmethod
    def _subs(self, x: Tensor, y: Tensor) -> Tensor:
        "The implementation for ``.sub``."
        ...

    def plot(self, canvas: Canvas) -> None:
        "Plot method for ``LinearEq``."

        xs = canvas.xs()
        start_x = xs[0]
        end_x = xs[-1]

        start_y, end_y = self.solve([start_x, end_x])

        canvas.line(Vec2d(start_x, start_y), Vec2d(end_x, end_y), color="blue")


@dcls.dataclass(frozen=True)
class StandardForm(LinearEq):
    """
    The standard form ax + by + c = 0
    """

    a: float
    """
    The x coefficient.
    """

    b: float
    """
    The y coefficient.
    """

    c: float
    """
    The constant term.
    """

    @typing.override
    def _solve(self, x: Tensor) -> Tensor:
        # y = (ax + c) / -b
        return (self.a * x + self.c) / -self.b

    @typing.override
    def _subs(self, x: Tensor, y: Tensor) -> Tensor:
        return self.a * x + self.b * y + self.c


@dcls.dataclass(frozen=True)
class SlopeInterceptForm(LinearEq):
    """
    An equation mx + b - y = 0.
    """

    m: float
    """
    The slop of the current equation.
    """

    b: float
    """
    The line passes over (0, b).
    """

    @typing.override
    def _solve(self, x: Tensor) -> Tensor:
        return self.m * x + self.b

    @typing.override
    def _subs(self, x: Tensor, y: Tensor) -> Tensor:
        return self.m * x + self.b - y


@dcls.dataclass(frozen=True)
class InterceptForm(LinearEq):
    """
    A linear equation represented by the intercepts.

    The equation is represented as x/a + y/b - 1 = 0
    """

    a: float
    """
    The line passes over (a, 0).
    """

    b: float
    """
    The line passes over (0, b).
    """

    @typing.override
    def _solve(self, x: Tensor) -> Tensor:
        # y = -b (x/a - 1)
        return self.b - x * self.b / self.a

    @typing.override
    def _subs(self, x: Tensor, y: Tensor) -> Tensor:
        return x / self.a + y / self.b - 1


def _promote_to_tensor(x: int | float | Sequence[int | float] | Tensor, /) -> Tensor:
    def convert() -> Tensor:
        if isinstance(x, Tensor):
            return x

        if isinstance(x, int | float):
            return torch.tensor(x)

        # Check this last because it can be expensive.
        if _is_seq_of_numbers(x):
            return torch.tensor(x)

        raise TypeError(f"Unsupported type: {type(x)=}")

    ans = convert()
    assert ans.ndim in [0, 1], f"Should be a 0D or 1D Tensor. Got {ans.ndim}D"
    return ans


def _is_seq_of_numbers(x: object) -> TypeIs[Sequence[int | float]]:
    return isinstance(x, Sequence) and all(isinstance(v, int | float) for v in x)
