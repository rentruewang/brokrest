# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API ``Topo``, representing a set of shapes."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import ClassVar, Self, TypeIs

import numpy as np
import torch
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray
from tensordict import TensorDict
from torch import Size, Tensor
from torch._tensor import Tensor

from brokrest.plotting import Canvas, Displayable

if typing.TYPE_CHECKING:
    from .rects import Box

__all__ = ["Topo", "Shape"]


@dcls.dataclass
class Topo(ABC):
    """
    A set of topologies.
    """

    data: TensorDict
    """
    The backing data.
    """

    KEYS: ClassVar[tuple[str, ...]]
    """
    The keys a class accepts, in order.
    """

    def __post_init__(self) -> None:
        self._validate_keys()
        self._ensure_shapes()
        self._order_self()

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __repr__(self):
        return repr(self.data)

    @typing.final
    def __len__(self) -> int:
        return len(self.data)

    @typing.overload
    def __getitem__(self, idx: str) -> Tensor: ...

    @typing.overload
    def __getitem__(self, idx: list[str]) -> TensorDict: ...

    @typing.overload
    def __getitem__(self, idx: int | slice | list[int] | NDArray | Tensor) -> Self: ...

    @typing.final
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._getitem_str(idx)

        if _list_of_str(idx):
            return self._getitem_list_str(idx)

        if isinstance(idx, int | slice | Tensor):
            return self._getitem_rows(idx)

        # Numpy compatible types (more expensive to construct).
        if np.isdtype((arr := np.array(idx)).dtype, "integral"):
            return self._getitem_rows(arr)

        raise ValueError(f"{type(idx)=} is not supported!")

    def _getitem_str(self, idx: str) -> Tensor:
        return self.data[idx]

    def _getitem_list_str(self, idx: list[str]) -> TensorDict:
        return self.data.select(*idx)

    def _getitem_rows(self, idx: int | slice | list[int] | NDArray | Tensor) -> Self:
        return type(self)(self.data[idx])

    def _validate_keys(self) -> None:
        "Check if all keys specified are present."

        data_keys: Iterable[str] = self.data.keys()

        if set(self.keys()).difference(data_keys):
            raise KeyError(
                f"Required keys: {self.keys()=}, but self.data has keys: {data_keys}"
            )

    def _ensure_shapes(self) -> None:
        "Check if shapes are valid."

        # This is s.t. we don't need to manually set ``batch_size`` or ``shape``.
        self.data.auto_batch_size_()

        # Must be 0D or 1D tensors.
        if self.ndim not in [0, 1]:
            raise ValueError(f"Tensors must be 0D or 1D. Got {self.ndim}D.")

        auto_shape = torch.broadcast_shapes(*map(lambda t: t.shape, self.values()))
        if auto_shape != self.batch_size:
            raise ValueError(
                f"Discovered batch size: {self.batch_size} "
                f"should match the broadcasted shape: {auto_shape}."
            )

    def _order_self(self) -> None:
        "Sort according to ``argsort``."

        ordering = self.ordering()

        # Do nothing if ``self.ordering() is None``, or if it's an instance not sequence.
        if ordering is None or self.ndim == 0:
            return

        if len(ordering) != len(self):
            raise ValueError(
                " ".join(
                    [
                        f"`ordering` should be a permutation of 0-{len(self)=}.",
                        f"Got a {ordering.ndim}D array with shape {ordering.shape}.",
                    ]
                )
            )

        ordered_index = torch.argsort(ordering)
        self.data = self.data[ordered_index]

    def keys(self) -> tuple[str, ...]:
        """
        The underlying ``Tensor``s backing the current topology.
        All the tensors yielded should have the same shape.

        Yields:
            Tensors with the same shape.
        """

        return self.KEYS

    @typing.final
    def values(self) -> Iterator[Tensor]:
        for key in self.keys():
            yield self[key]

    @typing.final
    def items(self) -> Iterator[tuple[str, Tensor]]:
        for key in self.keys():
            yield key, self[key]

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

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def batch_size(self) -> Size:
        return self.data.batch_size

    def numel(self) -> int:
        return self.data.numel()

    def to_dict(self):
        return self.data.to_dict()

    def item(self):
        return self.data.item()

    def to(self, device: str) -> Self:
        self.data = self.data.to(device)
        return self

    @classmethod
    def init(cls, **tensors: Tensor) -> Self:
        """
        Construct a ``TensorDict`` from input, and set ``self.data`` to it.
        Broadcasts the input ``Tensor``s to the same shape before constructing
        the ``TensorDict``, guaranteeing that the generated ``TensorDict``
        would always discover the biggest common batch size in shared axeses.

        Returns:
            An instance of ``Self`` (depends on which class calls this method).
        """

        tensors = broadcast_tensor_dict(tensors)
        try:
            return cls(data=TensorDict(tensors))
        except KeyError as ke:
            raise TypeError(
                f"Expected {[*tensors.keys()]=} to contain all keys from {cls.KEYS=}."
            ) from ke


class Shape(Displayable, Topo, ABC):
    """
    A topo set representing shapes that have clear boundaries.
    """

    @typing.override
    def draw(self, canvas: Canvas, /) -> None:
        """
        Populate the canvas with ``bokeh``, filter based on viewbox (``self.outer()``).
        """

        selected = self

        # Get the bounding box of ``self``, and get rid of points not in the box.
        box = self.outer()
        visible_topos = box.visible(canvas.window)
        selected = selected[visible_topos]

        selected._draw(canvas.figure)

    @abc.abstractmethod
    def _draw(self, figure: Figure, /) -> None:
        """
        The implementation of ``draw``.

        Args:
            figure: A ``bokeh`` figure.
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
        "Implementation of ``outer``."

        ...


def broadcast_tensor_dict(items: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    Broadcast the tensors in a mapping from string to tensors to the same shape.
    """

    keys = list(items.keys())
    vals = [items[k] for k in keys]
    return {k: v for k, v in zip(keys, torch.broadcast_tensors(*vals))}


def _list_of_str(obj: object) -> TypeIs[list[str]]:
    "Check if ``obj`` is ``list[str]``. Expensive."

    return isinstance(obj, list) and all(isinstance(elem, str) for elem in obj)
