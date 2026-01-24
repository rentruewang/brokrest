# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of rulers reliant on linear regressions."

import dataclasses as dcls
import typing
from typing import NamedTuple

import torch
from torch import Tensor, linalg

from brokrest.topos import Line, Point

from .rulers import Ruler

__all__ = ["LineReg"]


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
        return Line.init(m=m, b=b)


def _bias_expand(x: Tensor) -> Tensor:
    ones = torch.ones_like(x)
    return torch.stack([x, ones], dim=-1)


def _expand(x: Tensor) -> Tensor:
    return x[..., None]


class _SlopeAndBias(NamedTuple):
    m: Tensor
    b: Tensor


def _bias_parse(sol: Tensor) -> _SlopeAndBias:
    return _SlopeAndBias(m=sol[0], b=sol[1])


def _parse(sol: Tensor) -> _SlopeAndBias:
    m = sol[0]
    return _SlopeAndBias(m=m, b=torch.zeros_like(m))
