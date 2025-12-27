# Copyright (c) The BrokRest Authors - All Rights Reserved

"Points on a Cartesian plane."

import dataclasses as dcls
import math
from typing import Callable, Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["Vec2d"]


@dcls.dataclass(frozen=True)
class Vec2d:
    """
    A vector that can be represented as pair on the cartesian plane.
    """

    x: float
    "The ``x`` value."

    y: float
    "The ``y`` value."

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: Self | float) -> Self:
        return self.__binary_arithmetic(other, lambda l, r: l + r)

    def __sub__(self, other: Self | float) -> Self:
        return self.__binary_arithmetic(other, lambda l, r: l - r)

    def __mul__(self, other: Self | float) -> Self:
        return self.__binary_arithmetic(other, lambda l, r: l * r)

    def __truediv__(self, other: Self | float) -> Self:
        return self.__binary_arithmetic(other, lambda l, r: l / r)

    def __floordiv__(self, other: Self | float) -> Self:
        return self.__binary_arithmetic(other, lambda l, r: l // r)

    def __array__(self) -> NDArray:
        return np.array([self.x, self.y])

    def __binary_arithmetic(
        self,
        other: Self | float,
        op: Callable[[float, float], float],
    ) -> Self:
        if isinstance(other, Vec2d):
            return type(self)(x=op(self.x, other.x), y=op(self.y, other.y))

        if isinstance(other, float):
            return type(self)(x=op(self.x, other), y=op(self.y, other))

        # Not sure how to handle others.
        return NotImplemented

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
