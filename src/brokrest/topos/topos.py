# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API `Topo`, representing a set of shapes."

import abc
import typing
from collections import abc as cabc

import numpy as np
import tensordict as td
import torch
from bokeh import plotting
from numpy import typing as npt

from brokrest.plotting import Canvas, Displayable
from brokrest.tds import TensorClass

if typing.TYPE_CHECKING:
    from .rects import Box

__all__ = ["Topo"]


class ArrayOf[T](typing.Protocol):
    def __len__(self) -> int: ...

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...
    @typing.overload
    def __getitem__(self, idx: slice | list[int] | npt.NDArray[np.int_]) -> T: ...


class Topo(TensorClass, Displayable, abc.ABC):
    """
    A set of topologies.

    This is backed by `td.tensorclass`.
    If `.ndim == 0`, this is a single instance and you can call `item()` on it.
    """

    def __post_init__(self) -> None:
        self._setup_batch_size()
        self._sort_by_value()

    def _setup_batch_size(self):
        # Make all equal size.
        _ = self.auto_batch_size_()

        # Check if shapes are valid.
        shapes = [t.shape for t in self.values()]
        auto_shape = torch.broadcast_shapes(*shapes)
        if auto_shape != self.shape:
            raise ValueError(
                f"Discovered batch size: {self.shape} "
                f"should match the broadcasted shape: {auto_shape}."
            )

    @typing.no_type_check
    def _sort_by_value(self) -> None:
        "Sort according to `ordering`."

        if (ordering := self.ordering()) is NotImplemented:
            return

        # This is OK. Both are scalars.
        if self.ndim == ordering.ndim == 0:
            return

        if not (self.ndim == ordering.ndim == 1):
            raise NotImplementedError(
                f"Ndim != 1 is not yet supported {self.ndim=}, {ordering.ndim=}."
            )

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

    def ordering(self) -> torch.Tensor:
        """
        Return the argsort of the current `Topo`.

        If the collection doesn't need to be ordered, return `NotImplemented`.

        Returns:
            A 1D `torch.Tensor` of shape [len(self)],
            whose elements are permutation of `range(len(self))`,
            or `NotImplemented` if ordering doesn't exist.
        """

        return NotImplemented

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

        raise NotImplementedError

    def outer(self) -> "Box":
        """
        Get the outer boundary of the current topology,
        s.t. we can easily filter, with vector / GPU operations,
        what to draw and what not to draw.

        Returns:
            The viewport of the underlying boxes.
        """

        # Perhaps this is not defined.
        if (outer := self._outer()) is NotImplemented:
            return NotImplemented

        if (ob := outer.shape) != (sb := self.shape):
            raise ValueError(
                f"The batch size yielded from `outer`'s implementation: {ob} "
                f"is not the same as self: {sb}."
            )

        return outer

    def _outer(self) -> "Box":
        "Implementation of `outer`."

        return NotImplemented

    def tensor(self) -> torch.Tensor:
        values = list(self.values())
        return torch.stack(values)

    @classmethod
    def from_dict(cls, items: cabc.Mapping[str, torch.Tensor]) -> typing.Self:
        broadcasted = _broadcast_tensor_dict(items)
        return cls(**broadcasted)


def _broadcast_tensor_dict(
    items: cabc.Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Broadcast the tensors in a mapping from string to tensors to the same shape.
    """

    keys = list(items.keys())
    vals = [items[k] for k in keys]
    return {k: v for k, v in zip(keys, torch.broadcast_tensors(*vals))}
