# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of rulers reliant on linear regressions."

import dataclasses as dcls
import math
import typing

import numpy as np
from numpy import linalg

from brokrest.topos import Line, Point
from brokrest.typing import FloatArray

from .rulers import Ruler

__all__ = ["LineReg"]


@dcls.dataclass
class LineReg(Ruler):
    """
    A linear regression solver that generates a `Line` over given points.
    """

    bias: bool
    """
    Whether or not the `Line` is passing over the origin.
    """

    @typing.override
    def __call__(self, points: Point, /) -> Line:
        expand = _bias_expand if self.bias else _expand
        parse = _bias_parse if self.bias else _parse

        x = expand(points.x)
        sol, *_ = linalg.lstsq(x, points.y)
        m, b = parse(sol)
        return Line.slope_intercept(m=m, b=b)


@typing.no_type_check
def pinned_linear_regression(points: Point, pins: list[int]) -> Line:
    """
    Linear regression on points, with some pins applied.
    Each pin would trigger a new linear regression (in parallel).
    """

    point = points[pins]
    points = points[..., None] - point[None, ...]
    linreg = LineReg(bias=False)
    line = linreg(points)
    return shift_line_on(line, point)


def shift_line_on(line: Line, point: Point) -> Line:
    "Shift the line to fit a point. The output would have a new dimension matching points."

    shifts = (point.y - line.solve_y(point.x)).flatten()
    m = line.slope[..., None]
    b = line.y_intercept[..., None] + shifts
    return Line.slope_intercept(m=m, b=b)


def shift_line_percentage(line: Line, point: Point, ratio: float) -> Line:
    values = line.subs(point)
    ordered = values.argsort()
    num_items = math.floor(ratio * len(line))
    selected = point[ordered[num_items]]
    return shift_line_on(line, selected)


def boundary_rotate_linereg(points: Point, /) -> Line:
    line = boundary_linereg(points)
    argmin, argmax = arg_minmax_for_line(line, points)
    return pinned_linear_regression(points, [argmin, argmax])


def boundary_linereg(points: Point, /) -> Line:
    """
    Do linear regression, then shift to top and bottom boundaries.
    """

    regression_line = LineReg(bias=True)(points)
    argmin, argmax = arg_minmax_for_line(regression_line, points)
    minmax = points[[argmin, argmax]]
    return shift_line_on(regression_line, minmax).flatten()


def arg_minmax_for_line(line: Line, points: Point) -> tuple[int, int]:
    value = line.subs(points)
    maximum = int(np.argmax(value))
    minimum = int(np.argmin(value))
    return minimum, maximum


def _bias_expand(x: FloatArray) -> FloatArray:
    ones = np.ones_like(x)
    return np.stack([x, ones], axis=-1)


def _expand(x: FloatArray) -> FloatArray:
    return x[..., None]


class _SlopeAndBias(typing.NamedTuple):
    m: FloatArray
    b: FloatArray


def _bias_parse(sol: np.ndarray) -> _SlopeAndBias:
    return _SlopeAndBias(m=sol[0], b=sol[1])


def _parse(sol: np.ndarray) -> _SlopeAndBias:
    m = sol[0]
    return _SlopeAndBias(m=m, b=np.zeros_like(m))
