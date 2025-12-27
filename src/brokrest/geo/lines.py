# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import dataclasses as dcls
from typing import Protocol


class LinearEq(Protocol):
    pass


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


@dcls.dataclass(frozen=True)
class SlopeInterceptForm(LinearEq):
    """
    An equation y = mx + b.
    """

    m: float
    """
    The slop of the current equation.
    """

    b: float
    """
    The line passes over (0, b).
    """


class InterceptForm(LinearEq):
    """
    A linear equation represented by the intercepts.

    The equation is represented as x/a + y/b = 1
    """

    a: float
    """
    The line passes over (a, 0).
    """

    b: float
    """
    The line passes over (0, b).
    """
