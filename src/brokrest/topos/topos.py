# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API ``Topo``, representing a set of shapes."

import abc
import typing
from abc import ABC
from collections.abc import Iterator
from typing import Self, TypeAlias

from bokeh.plotting import figure as Figure
from numpy.typing import NDArray
from tensordict import TensorClass
from torch import Size, Tensor
from torch.types import Number

from brokrest.plotting import Canvas

if typing.TYPE_CHECKING:
    from .geo import Box


class Topo(TensorClass, ABC):
    """
    A set of topologies.
    """

    # The "type stubs" of ``TensorDict`` as ``Topo`` is a ``TensorClass``.
    if typing.TYPE_CHECKING:

        def __len__(self) -> int: ...

        IndexType: TypeAlias = int | slice | list[int] | NDArray | Tensor
        """
        The index that is accepted.
        Using a ``TypeAlias`` s.t. it can be reused in other ``typing.TYPE_CHECKING`` blocks.
        """

        def __getitem__(self, idx: IndexType) -> Self: ...

        @property
        def batch_size(self) -> Size: ...

        @property
        def ndim(self) -> int: ...

        def to_dict(self) -> dict[str, Tensor]: ...

    def __post_init__(self) -> None:
        self._ensure_shapes()
        self._order_self()

    @typing.no_type_check
    def _ensure_shapes(self) -> None:
        "Check if shapes are valid."
        if not all(isinstance(t, Tensor) for t in self.tensors()):
            raise TypeError("Data should all be tensors.")

        # This is s.t. we don't need to manually set ``batch_size`` or ``shape``.
        self.auto_batch_size_()

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

    def item(self) -> dict[str, Number]:
        """
        Convert ``ItemSet`` with 1 element to an ``Item``.

        Raises:
            ValueError: If the number of items is not 1.

        Returns:
            Item: _description_
        """

        if (elems := len(self)) != 1:
            cls_name = type(self).__qualname__
            raise ValueError(
                f"An ``{cls_name}`` with {elems} elements cannot call `item()`."
            )

        return {k: v.item() for k, v in self.to_dict().items()}

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

    @typing.override
    def draw(self, canvas: Canvas, /) -> None:
        """
        Populate the canvas with ``bokeh``, filter based on viewbox (``self.outer()``).
        """

        # Get the bounding box of ``self``.
        box = self.outer()
        visible_topos = box.visible(canvas.window)
        filtered = self[visible_topos]
        filtered._draw(canvas.figure)

    @abc.abstractmethod
    def _draw(self, figure: Figure, /) -> None:
        """
        The implementation of ``draw``.

        Args:
            figure: A ``bokeh`` figure.
        """

        ...
