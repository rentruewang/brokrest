# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API `Topo`, representing a set of shapes."

import abc
import typing
from abc import ABC
from typing import TypeIs

import torch
from bokeh import plotting
from torch import Tensor

from brokrest.plotting import Canvas, Displayable
from brokrest.tds import TensorClass

if typing.TYPE_CHECKING:
    from .rects import Box

__all__ = ["Topo", "Shape"]


class Topo(TensorClass, ABC):
    """
    A set of topologies.
    """

    def __post_init__(self) -> None:
        # This is s.t. we don't need to manually set `batch_size` or `shape`.
        self.auto_batch_size_()

        self._ensure_shapes()
        self._sort_by_key()

    def _ensure_shapes(self) -> None:
        # Check if shapes are valid.

        auto_shape = torch.broadcast_shapes(*map(lambda t: t.shape, self.values()))
        if auto_shape != self.batch_size:
            raise ValueError(
                f"Discovered batch size: {self.batch_size} "
                f"should match the broadcasted shape: {auto_shape}."
            )

    @typing.no_type_check
    def _sort_by_key(self) -> None:
        "Sort according to `argsort`."

        ordering = self.sort_key()

        # Do nothing if `self.ordering() is None`, or if it's an instance not sequence.
        if ordering is None or self.ndim == 0:
            return

        if len(ordering) != len(self):
            raise ValueError(
                " ".join(
                    [
                        f"`Topo.sort_key()` should be a permutation of 0-{len(self)=}.",
                        f"Got a {ordering.ndim}D array with shape {ordering.shape}.",
                    ]
                )
            )

        ordered_index = torch.argsort(ordering)

        for key in self.keys():
            setattr(self, key, getattr(self, key)[ordered_index])

    def sort_key(self) -> Tensor | None:
        """
        Return the argsort of the current `Topo`.

        If the collection doesn't need to be ordered, return `NotImplemented`.

        Returns:
            A 1D `Tensor` of shape [len(self)],
            whose elements are permutation of `range(len(self))`,
            or `NotImplemented` if ordering doesn't exist.
        """

        return None


class Shape(Displayable, Topo, ABC):
    """
    A topo set representing shapes that have clear boundaries.
    """

    @typing.override
    def draw(self, canvas: Canvas, /) -> None:
        """
        Populate the canvas with `bokeh`, filter based on viewbox (`self.outer()`).
        """

        selected = self

        # Get the bounding box of `self`, and get rid of points not in the box.
        box = self.outer()
        visible_topos = box.visible(canvas.window)
        selected = selected[visible_topos]

        selected._draw(canvas.figure)

    @abc.abstractmethod
    def _draw(self, figure: plotting.figure, /) -> None:
        """
        The implementation of `draw`.

        Args:
            figure: A `bokeh` figure.
        """

        ...

    def outer(self) -> "Box":
        """
        Get the outer boundary of the current topology,
        s.t. we can easily filter, with vector / GPU operations,
        what to draw and what not to draw.

        Returns:
            The viewport of the underlying boxes.
        """

        outer = self._outer()

        if (ob := outer.batch_size) != (sb := self.batch_size):
            raise ValueError(
                f"The batch size yielded from `outer`'s implementation: {ob} "
                f"is not the same as self: {sb}."
            )

        return outer

    @abc.abstractmethod
    def _outer(self) -> "Box":
        "Implementation of `outer`."

        ...


def broadcast_tensor_dict(items: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Broadcast the tensors in a mapping from string to tensors to the same shape.
    """

    keys = list(items.keys())
    vals = [items[k] for k in keys]
    return {k: v for k, v in zip(keys, torch.broadcast_tensors(*vals))}


def _list_of_str(obj: object) -> TypeIs[list[str]]:
    "Check if `obj` is `list[str]`. Expensive."

    return isinstance(obj, list) and all(isinstance(elem, str) for elem in obj)
