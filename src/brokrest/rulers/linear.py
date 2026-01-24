# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of rulers reliant on linear regressions."

import dataclasses as dcls
import typing

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

        x = points.x
        y = points.y

        if self.bias:
            ones = torch.ones_like(x)
            x = torch.stack([x, ones], dim=-1)
        else:
            x = x[..., None]

        sol: Tensor = linalg.lstsq(x, y).solution

        if self.bias:
            return Line.init(m=sol[0], b=sol[1])
        else:
            return Line.init(m=sol[0], b=torch.zeros_like(sol[0]))
