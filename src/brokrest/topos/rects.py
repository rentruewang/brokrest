# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of shapes that can be represented as a 4 tuple."

import typing
from abc import ABC

import torch
from bokeh.plotting import figure as Figure
from torch import Tensor

from brokrest.plotting import Window

from .topos import Topo

__all__ = ["Rect", "Box", "Segment"]


class Rect(Topo, ABC):
    """
    A tuple with 4 values.

    If ``batch_size`` is not 1, all ``x_0``, ``x_1``, ``y_0``, ``y_1`` need to be the same shape.
    """

    x_0: Tensor
    "The left side."

    x_1: Tensor
    "The right side."

    y_0: Tensor
    "The top side."

    y_1: Tensor
    "The bottom side."

    @typing.override
    def tensors(self):
        yield self.x_0
        yield self.x_1
        yield self.y_0
        yield self.y_1


class Box(Rect):
    """
    A box with 4 sides.
    """

    def __post_init__(self):
        super().__post_init__()

        if not torch.all(self.width >= 0):
            raise ValueError("Box should not have negative width.")

        if not torch.all(self.height >= 0):
            raise ValueError("Box should not have negative height.")

    @property
    def bottom(self) -> Tensor:
        "The bottom of the box. Alias of ``y_0``."
        return self.y_0

    @property
    def top(self) -> Tensor:
        "The top of the box. Alias of ``y_1``."
        return self.y_1

    @property
    def left(self) -> Tensor:
        "The left of the box. Alias of ``x_0``."
        return self.x_0

    @property
    def right(self) -> Tensor:
        "The right of the box. Alias of ``x_1``."
        return self.x_1

    @property
    def width(self) -> Tensor:
        "The width of the ``Box``es."
        return self.x_1 - self.x_0

    @property
    def height(self) -> Tensor:
        "The height of the ``Box``es."
        return self.y_1 - self.y_0

    @property
    def area(self) -> Tensor:
        "The area of the ``Box``es."
        return self.width * self.height

    @typing.override
    def _outer(self):
        return self

    @typing.override
    def _draw(self, figure: Figure):
        _ = figure.rect(
            x=self.x_0.numpy(),
            y=self.y_0.numpy(),
            width=self.width.numpy(),
            height=self.height.numpy(),
        )

    def visible(self, window: Window) -> Tensor:
        """
        Return a boolean tensor, of whether ``self`` is visible in the view box or not.

        Args:
            window: The view port to determine.

        Returns:
            A boolean tensor the same length as ``self``.
        """

        horiz = _segment_visible(
            start=self.left,
            end=self.right,
            x=window.left,
            y=window.right,
        )
        verti = _segment_visible(
            start=self.bottom,
            end=self.top,
            x=window.bottom,
            y=window.top,
        )

        # Both horizontally and vertically visible.
        return horiz & verti


def _segment_visible(start: Tensor, end: Tensor, x: float, y: float) -> Tensor:
    """
    Try to see if segment [start, end] is visible in viewport [x, y], vectorized.s
    """

    result_shape = start.shape
    assert end.shape == start.shape

    def _ordered(*ordered: Tensor):
        "The tensors are ordered."

        answer = torch.ones(result_shape).bool()
        for smaller, larger in zip(ordered[:-1], ordered[1:]):
            answer &= smaller <= larger
        return answer

    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    # Let's laid it out on an axis.
    # the line is visible with one of the conditions:
    ans = torch.zeros(result_shape).bool()

    # start - x - end - y
    ans |= _ordered(start, x_tensor, end, y_tensor)

    # start - x - y - end
    ans |= _ordered(start, x_tensor, y_tensor, end)

    # x - start - y - end
    ans |= _ordered(x_tensor, start, y_tensor, end)

    # x - start - end - y
    ans |= _ordered(x_tensor, start, end, y_tensor)

    return ans


class Segment(Rect):
    def __post_init__(self):
        super().__post_init__()

        # Argsort according to ``x_0``.
        ordered = torch.argsort(self.x_0)

        self.x_0 = self.x_0[ordered]
        self.x_1 = self.x_1[ordered]
        self.y_0 = self.y_0[ordered]
        self.y_1 = self.y_1[ordered]

    @property
    def start(self) -> Tensor:
        "The starting point of a segment."
        return torch.hstack([self.x_0, self.y_0])

    @property
    def end(self) -> Tensor:
        "The ending point of a segment."
        return torch.hstack([self.x_1, self.y_1])

    @property
    def left(self):
        "The ``min(x)``."
        return torch.minimum(self.x_0, self.x_1)

    @property
    def right(self):
        "The ``max(x)``."
        return torch.maximum(self.x_0, self.x_1)

    @property
    def bottom(self):
        "The ``min(y)``."
        return torch.minimum(self.y_0, self.y_1)

    @property
    def top(self):
        "The ``max(y)``."
        return torch.maximum(self.y_0, self.y_1)

    @typing.override
    def _outer(self):
        return Box(x_0=self.left, x_1=self.right, y_0=self.bottom, y_1=self.top)

    @typing.override
    def _draw(self, figure: Figure):
        _ = figure.segment(
            x0=self.x_0.numpy(),
            x1=self.x_1.numpy(),
            y0=self.y_0.numpy(),
            y1=self.y_1.numpy(),
        )
