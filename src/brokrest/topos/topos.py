# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API `Topo`, representing a set of shapes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from bokeh import plotting

from brokrest.plotting import Canvas, Displayable
from brokrest.tds import TensorClass, tensorclass

if typing.TYPE_CHECKING:
    from .rects import Box

__all__ = [
    "Topo",
    "TopoHandler",
    "TopoHandlerProxy",
    "register_topo_handler",
    "enabled_topo_handlers",
]


@tensorclass
class Topo(TensorClass, Displayable, abc.ABC):
    """
    A set of topologies.

    This is backed by `td.tensorclass`.
    If `.ndim == 0`, this is a single instance and you can call `item()` on it.
    """

    def __init_subclass__(cls) -> None:
        # We do not want subclasses to define their `__post_init__`,
        # to ensure that `_TOPO_HANDLERS` have the initialized `Topo`.
        if "__post_init__" in cls.__dict__:
            raise ValueError(
                f"Sublcass should not define `__post_init__`, but {cls=} has it."
            )

    @typing.final
    def __post_init__(self) -> None:
        self._setup_batch_size()
        self._sort_by_value()
        self._checks()

        # This is done last for sure, since subclasses cannot have `__post_init__` defined.
        self._call_handlers()

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

    def _checks(self) -> None:
        return

    def _call_handlers(self):
        for handler in _TOPO_HANDLERS:
            handler(self)

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


class TopoHandler(typing.Protocol):
    __name__: str

    def __call__(self, topo: Topo, /) -> None: ...


@dcls.dataclass(frozen=True)
class TopoHandlerProxy:
    """
    The proxy object for handler, s.t. we can make priting easier,
    compared to using a `ctxl.contextmanager` closure directly.
    """

    handler: TopoHandler
    "The handler that is wrapped."

    @typing.override
    def __repr__(self) -> str:
        return f"TopoHandler<{self.handler.__name__}>"

    @ctxl.contextmanager
    def __call__(self) -> cabc.Generator[typing.Self]:
        _TOPO_HANDLERS.append(self.handler)
        try:
            yield self
        finally:
            _ = _TOPO_HANDLERS.pop()


def register_topo_handler(handler: TopoHandler, /):
    """
    Register a `TopoHandler` to call on `Topo` once initialization is done.
    """

    return TopoHandlerProxy(handler)


def enabled_topo_handlers():
    """
    Get all the currently enabled handlers (under the context managers) in a list,
    wrapped in their proxies. Modifying this list doesn't change global handlers stack.
    """

    return [TopoHandlerProxy(handler) for handler in _TOPO_HANDLERS]


_TOPO_HANDLERS: list[TopoHandler] = []


def _broadcast_tensor_dict(
    items: cabc.Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Broadcast the tensors in a mapping from string to tensors to the same shape.
    """

    keys = list(items.keys())
    vals = [items[k] for k in keys]
    return {k: v for k, v in zip(keys, torch.broadcast_tensors(*vals))}
