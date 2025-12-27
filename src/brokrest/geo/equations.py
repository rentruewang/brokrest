# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import abc
import dataclasses as dcls
from typing import Protocol
import typing
from abc import ABC


class LinearEq(ABC):
    """
    A linear equation.
    """

    @abc.abstractmethod
    def solve(self, x: float) -> float:
        """
        Get the y value of the line equation when x is given.

        Args:
            x: The x value.

        Returns:
            The y value.
        """

        ...

    @abc.abstractmethod
    def subs(self, x: float, y: float) -> float:
        """
        The linear equation assumes that RHS = 0.
        This method applies x, y to the LHS, and get the value of RHS.

        If this method returns 0, the point is on the line.
        If 2 points (a, b), (c, d) have the same signs (both positive or both negative)
        after substituted into the equation, they are on the same side of the line.
        """

        ...


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
    def solve(self, x: float) -> float:
        return (self.a * x + self.c) / self.b

    @typing.override
    def subs(self, x: float, y: float) -> float:
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
    def solve(self, x: float) -> float:
        return self.m * x + self.b

    @typing.override
    def subs(self, x: float, y: float) -> float:
        return self.m * x + self.b - y


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
    def solve(self, x: float) -> float:
        # y = -b (x/a - 1)
        return self.b - x * self.a / self.b

    @typing.override
    def subs(self, x: float, y: float) -> float:
        return x / self.a + y / self.b - 1
