# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of rulers reliant on linear regressions."

import dataclasses as dcls
import math
import typing

import torch
from torch import linalg

from brokrest import topos

from . import rulers

__all__ = ["LineReg"]


@dcls.dataclass
class LineReg(rulers.Ruler):
    """
    A linear regression solver that generates a `topos.Line` over given points.
    """

    bias: bool
    """
    Whether or not the `topos.Line` is passing over the origin.
    """

    @typing.override
    def __call__(self, points: topos.Point, /) -> topos.Line:
        expand = _bias_expand if self.bias else _expand
        parse = _bias_parse if self.bias else _parse

        x = expand(points.x)
        sol: torch.Tensor = linalg.lstsq(x, points.y).solution
        m, b = parse(sol)
        return topos.Line(m=m, b=b)


def pinned_linear_regression(points: topos.Point, *pin: int) -> topos.Line:
    """
    Linear regression on points, with some pins applied.
    Each pin would trigger a new linear regression (in parallel).
    """

    pin_idx = list(pin)
    point = points[pin_idx]
    points = points[..., None] - point[None, ...]
    linreg = LineReg(bias=False)
    line = linreg(points)
    return shift_line_on(line, point)


def shift_line_on(line: topos.Line, point: topos.Point) -> topos.Line:
    "Shift the line to fit a point. The output would have a new dimension matching points."

    shifts = (point.y - line.apply(point.x)).flatten()
    m = line.m[..., None]
    b = line.b[..., None] + shifts
    return topos.Line(m=m, b=b)


def shift_line_percentage(
    line: topos.Line, point: topos.Point, ratio: float
) -> topos.Line:
    values = line.subs(point)
    ordered = values.argsort()
    num_items = math.floor(ratio * len(line))
    selected = point[ordered[num_items]]
    return shift_line_on(line, selected)


def boundary_rotate_linereg(points: topos.Point, /) -> topos.Line:
    line = boundary_linereg(points)
    argmin, argmax = arg_minmax_for_line(line, points)
    return pinned_linear_regression(points, argmin, argmax)


def boundary_linereg(points: topos.Point, /) -> topos.Line:
    """
    Do linear regression, then shift to top and bottom boundaries.
    """

    regression_line = LineReg(bias=True)(points)
    argmin, argmax = arg_minmax_for_line(regression_line, points)
    minmax = points[[argmin, argmax]]
    return shift_line_on(regression_line, minmax).flatten()


def arg_minmax_for_line(line: topos.Line, points: topos.Point) -> tuple[int, int]:
    value = line.subs(points)
    maximum = int(torch.argmax(value).item())
    minimum = int(torch.argmin(value).item())
    return minimum, maximum


def _bias_expand(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(x)
    return torch.stack([x, ones], dim=-1)


def _expand(x: torch.Tensor) -> torch.Tensor:
    return x[..., None]


class _SlopeAndBias(typing.NamedTuple):
    m: torch.Tensor
    b: torch.Tensor


def _bias_parse(sol: torch.Tensor) -> _SlopeAndBias:
    return _SlopeAndBias(m=sol[0], b=sol[1])


def _parse(sol: torch.Tensor) -> _SlopeAndBias:
    m = sol[0]
    return _SlopeAndBias(m=m, b=torch.zeros_like(m))
