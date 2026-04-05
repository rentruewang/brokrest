# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of shapes that can be represented as a 4 tuple."

import abc
import typing

import torch
from bokeh import plotting

from brokrest.plotting import Window
from brokrest.tds import tensorclass

from .topos import Shape

__all__ = ["Rect", "Box", "Segment"]


@tensorclass
class Rect(Shape, abc.ABC):
    """
    A tuple with 4 values.

    If `batch_size` is not 1, all `x_0`, `x_1`, `y_0`, `y_1` need to be the same shape.
    """

    x_0: torch.Tensor
    "The left side."

    x_1: torch.Tensor
    "The right side."

    y_0: torch.Tensor
    "The top side."

    y_1: torch.Tensor
    "The bottom side."


@tensorclass
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
    def bottom(self) -> torch.Tensor:
        "The bottom of the box. Alias of `y_0`."
        return self.y_0

    @property
    def top(self) -> torch.Tensor:
        "The top of the box. Alias of `y_1`."
        return self.y_1

    @property
    def left(self) -> torch.Tensor:
        "The left of the box. Alias of `x_0`."
        return self.x_0

    @property
    def right(self) -> torch.Tensor:
        "The right of the box. Alias of `x_1`."
        return self.x_1

    @property
    def width(self) -> torch.Tensor:
        "The width of the `Box`es."
        return self.x_1 - self.x_0

    @property
    def height(self) -> torch.Tensor:
        "The height of the `Box`es."
        return self.y_1 - self.y_0

    @property
    def area(self) -> torch.Tensor:
        "The area of the `Box`es."
        return self.width * self.height

    @typing.override
    def _outer(self):
        return self

    @typing.override
    def _draw(self, figure: plotting.figure):
        _ = figure.rect(
            x=self.x_0.numpy(),
            y=self.y_0.numpy(),
            width=self.width.numpy(),
            height=self.height.numpy(),
        )

    def visible(self, window: Window) -> torch.Tensor:
        """
        Return a boolean tensor, of whether `self` is visible in the view box or not.

        Args:
            window: The view port to determine.

        Returns:
            A boolean tensor the same length as `self`.
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


def _segment_visible(
    start: torch.Tensor, end: torch.Tensor, x: float, y: float
) -> torch.Tensor:
    """
    Try to see if segment [start, end] is visible in viewport [x, y], vectorized.s
    """

    result_shape = start.shape
    assert end.shape == start.shape

    def _ordered(*ordered: torch.Tensor):
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


@tensorclass
class Segment(Rect):

    @typing.override
    def sort_key(self) -> torch.Tensor:
        return torch.argsort(self.x_0)

    @property
    def start(self) -> torch.Tensor:
        "The starting point of a segment."

        return torch.hstack([self.x_0, self.y_0])

    @property
    def end(self) -> torch.Tensor:
        "The ending point of a segment."

        return torch.hstack([self.x_1, self.y_1])

    @property
    def left(self):
        "The `min(x)`."

        return torch.minimum(self.x_0, self.x_1)

    @property
    def right(self):
        "The `max(x)`."

        return torch.maximum(self.x_0, self.x_1)

    @property
    def bottom(self):
        "The `min(y)`."
        return torch.minimum(self.y_0, self.y_1)

    @property
    def top(self):
        "The `max(y)`."
        return torch.maximum(self.y_0, self.y_1)

    @typing.override
    def _outer(self):
        return Box(x_0=self.left, x_1=self.right, y_0=self.bottom, y_1=self.top)

    @typing.override
    def _draw(self, figure: plotting.figure):
        _ = figure.segment(
            x0=self.x_0.numpy(),
            x1=self.x_1.numpy(),
            y0=self.y_0.numpy(),
            y1=self.y_1.numpy(),
        )
