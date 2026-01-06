# Copyright (c) The BrokRest Authors - All Rights Reserved

"The classes that are a stack of data."

import abc
import typing
from abc import ABC
from collections.abc import Iterator
from typing import Self, override

import torch
from bokeh.plotting import figure as Figure
from tensordict import TensorClass
from torch import Tensor

from brokrest.plotting import ViewPort

from .topos import Topo

__all__ = ["Group", "Rect", "Box", "Segment", "Candle"]


class Group(TensorClass, Topo, ABC):
    """
    A group of similar shapes, backed by ``Tensor``s.
    """

    def __post_init__(self) -> None:
        self._ensure_shapes()
        self._order_self()

    @typing.no_type_check
    def _ensure_shapes(self) -> None:
        "Check if shapes are valid."
        if not all(isinstance(t, Tensor) for t in self.tensors()):
            raise TypeError("Data should all be tensors.")

        if len(shapes := {t.shape for t in self.tensors()}) != 1:
            raise ValueError("Data must all have the same shapes.")

        # This is s.t. we don't need to manually set ``batch_size`` or ``shape``.
        [self.batch_size] = shapes

        # Must be 0D or 1D tensors.
        if self.ndim not in [0, 1]:
            raise ValueError(f"Tensors must be 0D or 1D. Got {self.ndim}D.")

    def _order_self(self) -> None:
        "Sort according to ``argsort``."

        ordering = self.ordering()

        # Do nothing if ``self.ordering() is None``.
        if ordering is None:
            return

        if ordering.ndim != 1 or len(ordering) != len(self):
            raise ValueError(
                " ".join(
                    [
                        f"`ordering` should be a 1D array, permutation of 0-{len(self)=}.",
                        f"Got a {ordering.ndim}D array with shape {ordering.shape}.",
                    ]
                )
            )

    @abc.abstractmethod
    def tensors(self) -> Iterator[Tensor]:
        """
        The underlying ``Tensor``s backing the current topology.
        All the tensors yielded should have the same shape.

        Yields:
            Tensors with the same shape.
        """

        ...

    def ordering(self) -> Tensor | None:
        """
        Return the argsort of the current ``Topology``.

        If the collection doesn't need to be ordered, return ``NotImplemented``.

        Returns:
            A 1D ``Tensor`` of shape [len(self)],
            whose elements are permutation of ``range(len(self))``,
            or ``NotImplemented`` if ordering doesn't exist.
        """

        return None


class Rect(Group, ABC):
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
    def _cut(self, vp: ViewPort, /) -> Self:
        see_left = self.x_0 < vp.right
        see_right = self.x_1 > vp.left
        see_bottom = self.y_0 < vp.top
        see_top = self.y_1 > vp.bottom

        indices = see_left | see_right | see_bottom | see_top
        return typing.cast(Self, self[indices])

    @typing.override
    def _draw(self, figure: Figure):
        _ = figure.rect(
            x=self.x_0.numpy(),
            y=self.y_0.numpy(),
            width=self.width.numpy(),
            height=self.height.numpy(),
        )


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
    def _cut(self, vp: ViewPort, /) -> Self:
        see_left = self.left < vp.right
        see_right = self.right > vp.left
        see_bottom = self.bottom < vp.top
        see_top = self.top > vp.bottom

        indices = see_left | see_right | see_bottom | see_top
        return typing.cast(Self, self[indices])

    @typing.override
    def _draw(self, figure: Figure):
        _ = figure.segment(
            x0=self.x_0.numpy(),
            x1=self.x_1.numpy(),
            y0=self.y_0.numpy(),
            y1=self.y_1.numpy(),
        )


class Candle(Group):
    """
    One single candle bar.
    """

    enter: Tensor
    """
    The entering position of this candle.
    """

    exit: Tensor
    """
    The exiting position of this candle.
    """

    low: Tensor
    """
    The minimum value of the candle.
    """

    high: Tensor
    """
    The maximum value of the candle.
    """

    start: Tensor
    """
    The starting time of the candle.
    """

    end: Tensor
    """
    The ending time of the candle.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        if torch.any(self.low > self.high):
            raise ValueError(
                f"Min value: {self.low} must be smaller than max value: {self.high}."
            )

        if torch.any(self.start > self.end):
            raise ValueError(
                f"Starting time {self.start} must be before than ending time: {self.end}."
            )
        self._check_value_in_range(self.enter, "entering")
        self._check_value_in_range(self.exit, "exiting")

    def _check_value_in_range(self, value: Tensor, desc: str) -> None:
        "Check if the given value is in the range [min, max]."

        if torch.any(self.low > value) or torch.any(self.high < value):
            message = " ".join(
                [
                    f"{desc.capitalize()} value: {self.enter},",
                    f"but min={self.low}, max={self.high}.",
                ]
            )
            raise ValueError(message)

    def continuous(self) -> bool:
        nexts: Self = self[1:]
        prevs: Self = self[:-1]
        return torch.allclose(nexts.start, prevs.end)

    @property
    def inc(self):
        "Is increasing."
        return (self.exit - self.enter) >= 0

    @property
    def dec(self):
        "Is decreasing."
        return (self.exit - self.enter) < 0

    @property
    def time(self):
        "The centered times."
        return (self.end + self.start) / 2

    @property
    def width(self):
        "The width for each candle."
        return self.end - self.start

    @typing.override
    def tensors(self):
        yield self.enter
        yield self.exit
        yield self.low
        yield self.high
        yield self.start
        yield self.end

    @override
    def _draw(self, figure: Figure):
        # The center bars for the candles.
        _ = figure.segment(
            x0=self.time.numpy(),
            y0=self.high.numpy(),
            x1=self.time.numpy(),
            y1=self.low.numpy(),
            color="black",
        )

        # The body of candles that are decreasing.
        _ = figure.vbar(
            x=self.time[self.dec].numpy(),
            width=self.width.numpy(),
            top=self.enter[self.dec].numpy(),
            bottom=self.exit[self.dec].numpy(),
            color="#eb3c40",
        )

        # The body of candles that are increasing.
        _ = figure.vbar(
            x=self.time[self.inc].numpy(),
            width=self.width.numpy(),
            top=self.enter[self.inc].numpy(),
            bottom=self.exit[self.inc].numpy(),
            fill_color="white",
            line_color="#49a3a3",
            line_width=2,
        )

    @typing.override
    def _cut(self, vp: ViewPort, /) -> Self:
        see_left = self.start < vp.right
        see_right = self.end > vp.left
        see_bottom = self.low < vp.top
        see_top = self.high > vp.bottom

        indices = see_left | see_right | see_bottom | see_top
        return typing.cast(Self, self[indices])
