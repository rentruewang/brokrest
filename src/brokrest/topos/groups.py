# Copyright (c) The BrokRest Authors - All Rights Reserved

"The classes that are a stack of data."

import abc
import typing
from abc import ABCMeta
from collections.abc import Iterator

import torch
from tensordict import TensorClass
from torch import Tensor
from typing import Self

__all__ = ["Group", "Rect", "Box", "Segment", "Candle"]


class Group(TensorClass, metaclass=ABCMeta):
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


class Rect(Group):
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

    @typing.override
    def tensors(self):
        yield self.enter
        yield self.exit
        yield self.low
        yield self.high
        yield self.start
        yield self.end
