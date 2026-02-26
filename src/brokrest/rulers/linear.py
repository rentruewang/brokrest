# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of rulers reliant on linear regressions."

import dataclasses as dcls
import math
import typing
from typing import NamedTuple

import torch
from torch import Tensor, linalg

from brokrest.topos import Line, Point

from .rulers import Ruler

__all__ = ["LineReg", "BoundaryRulerLineReg"]


@dcls.dataclass
class LineReg(Ruler):
    """
    A linear regression solver that generates a ``Line`` over given points.
    """

    bias: bool
    """
    Whether or not the ``Line`` is passing over the origin.
    """

    @typing.override
    def __call__(self, points: Point, /) -> Line:
        expand = _bias_expand if self.bias else _expand
        parse = _bias_parse if self.bias else _parse

        x = expand(points.x)
        sol: Tensor = linalg.lstsq(x, points.y).solution
        m, b = parse(sol)
        line = Line.init(m=m, b=b)
        assert line.batch_size == points.batch_size[:-1]
        return line


@dcls.dataclass(frozen=True)
class BoundaryRulerLineReg(Ruler):
    """
    Using linear regression to find the a good ``Ruler``,
    and shift to boundary to discover lines.
    """

    rotate: bool = False
    """
    Whether or not to perform rotation at the boundary.
    """

    shift_ratio: float = 0.0
    """
    Whether or not to allow ``ratio`` amount of points to be outside the boundary.
    Ratio is between 0-1.
    """

    def __post_init__(self):
        if not 0 <= self.shift_ratio <= 1:
            raise ValueError(f"{self.shift_ratio=} not in 0 - 1. Invalid.")

    @typing.override
    def __call__(self, points: Point, /) -> Line:

        line = boundary_linereg(points)
        argmin, argmax = arg_minmax_for_line(line, points)

        if self.rotate:
            line = pinned_linear_regression(points, torch.cat([argmin, argmax]))

        if self.shift_ratio:
            line = shift_line_ratio(
                line,
                point=points[[argmin, argmax]],
                ratio=self.shift_ratio / 2,
            )

        return line


def pinned_linear_regression(points: Point, pin: Tensor) -> Line:
    """
    Linear regression on points, with some pins applied.
    Each pin would trigger a new linear regression (in parallel).
    """

    assert pin.ndim == 1
    pinned = points[pin]
    biased_bcasted = points[None, ...] - pinned[..., None]
    linreg = LineReg(bias=False)
    line = linreg(biased_bcasted)
    assert len(biased_bcasted) == len(pin)
    return shift_line_on(line, pinned)


def shift_line_on(line: Line, point: Point) -> Line:
    "Shift the line to fit a point. The output would have a new dimension matching points."

    shifts = (point.y - line.apply(point.x)).flatten()
    m = line.m[..., None]
    b = line.b[..., None] + shifts
    return Line.init(m=m, b=b)


def shift_line_ratio(line: Line, point: Point, ratio: float) -> Line:
    "Shift line based on the ratio."

    values = line.subs(point)
    ordered = values.argsort(dim=-1)
    num_items = math.floor(ratio * len(line))
    selected = point[ordered[..., num_items]]
    return shift_line_on(line, selected)


def boundary_linereg(points: Point, /) -> Line:
    """
    Do linear regression, then shift to top and bottom boundaries.
    """

    regression_line = LineReg(bias=True)(points)
    argmin, argmax = arg_minmax_for_line(regression_line, points)
    minmax = points[torch.cat([argmin, argmax], dim=0)]
    return shift_line_on(regression_line, minmax).flatten()


def arg_minmax_for_line(line: Line, points: Point) -> tuple[Tensor, Tensor]:
    """
    Get the argmin, argmax of the points when substituted into (each) lines.

    Yields min, max tensors, each the size of ``line``.
    """

    value = line.subs(points)
    maximum = value.argmax(dim=-1)
    minimum = value.argmin(dim=-1)
    assert len(minimum) == len(maximum) == len(line)
    return minimum, maximum


def _bias_expand(x: Tensor) -> Tensor:
    ones = torch.ones_like(x)
    return torch.stack([x, ones], dim=-1)


def _expand(x: Tensor) -> Tensor:
    return x[..., None]


class _SlopeAndBias(NamedTuple):
    m: Tensor
    b: Tensor


def _bias_parse(sol: Tensor) -> _SlopeAndBias:
    return _SlopeAndBias(m=sol[..., 0, 0], b=sol[..., 1, 0])


def _parse(sol: Tensor) -> _SlopeAndBias:
    m = sol[..., 0, 0]
    return _SlopeAndBias(m=m, b=torch.zeros_like(m))
