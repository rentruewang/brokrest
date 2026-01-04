# Copyright (c) The BrokRest Authors - All Rights Reserved

"""Points on a Cartesian plane."""

from __future__ import annotations

import dataclasses as dcls
import functools
import math
import operator
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from typing import Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["Vec2d"]

_Real = (int, float)  # tuple for isinstance checks


def _binary_arithmetic_template(op: Callable[[_Real, _Real], _Real], /):
    """
    Template method for operators like +, -, *, /, // etc.

    Args:
        op: The operator performed elementwise that the method would do.

    Returns:
        A method.
    """

    @functools.wraps(op)
    def method(self: "Vec2d", other: "Vec2d | _Real") -> "Vec2d":
        if isinstance(other, Vec2d):
            return type(self)(x=op(self.x, other.x), y=op(self.y, other.y))

        if isinstance(other, _Real):
            return type(self)(x=op(self.x, other), y=op(self.y, other))

        # Not sure how to handle others.
        return NotImplemented

    return method


def flip(op: Callable[[_Real, _Real], _Real], /) -> Callable[[_Real, _Real], _Real]:
    """
    Flip left and right of the operation.
    E.g. lambda a, b: a - b would become lambda b, a: b - a
    """

    @functools.wraps(op)
    def flipped_op(a: _Real, b: _Real, /) -> _Real:
        return op(b, a)

    return flipped_op


@dcls.dataclass(frozen=True)
class Vec2d:
    """
    A vector that can be represented as pair on the cartesian plane.
    """

    x: _Real
    "The ``x`` value."

    y: _Real
    "The ``y`` value."

    def __iter__(self):
        yield self.x
        yield self.y

    def __array__(self) -> NDArray:
        return np.array([self.x, self.y])

    __add__ = __radd__ = _binary_arithmetic_template(operator.add)
    __sub__ = _binary_arithmetic_template(operator.sub)
    __rsub__ = _binary_arithmetic_template(flip(operator.sub))
    __mul__ = __rmul__ = _binary_arithmetic_template(operator.mul)
    __truediv__ = _binary_arithmetic_template(operator.truediv)
    __rtruediv__ = _binary_arithmetic_template(flip(operator.truediv))
    __floordiv__ = _binary_arithmetic_template(operator.floordiv)
    __rfloordiv__ = _binary_arithmetic_template(flip(operator.floordiv))
    __pow__ = _binary_arithmetic_template(operator.pow)
    __rpow__ = _binary_arithmetic_template(flip(operator.pow))

    def slope(self) -> float:
        """
        The "slope" of the current point to origin.
        """

        return self.y / self.x

    @classmethod
    def origin(cls) -> Self:
        """
        The origin point on the Cartesian plane.
        """

        return cls.cartesian(x=0, y=0)

    @classmethod
    def cartesian(cls, x: float, y: float) -> Self:
        """
        Constructor to create a ``Point`` from cartesian coordinate.

        Args:
            x: The x component.
            y: The y component.

        Returns:
            A point.
        """

        return cls(x=x, y=y)

    @classmethod
    def polar(cls, radius: float, theta: float) -> Self:
        """
        Construct a point with polar coordinate representation: (R, Theta).

        Args:
            radius: The distance to the origin.
            theta: The angle of the vector from the origin. In radian.

        Raises:
            ValueError: If the radius is < 0.

        Returns:
            A point.
        """
        if radius < 0:
            raise ValueError(f"Polar coordinate should have radius >= 0. Got {radius=}")

        theta %= 2 * math.pi

        return cls(x=radius * math.cos(theta), y=radius * math.sin(theta))
