# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API `Topo`, representing a set of shapes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from bokeh import plotting

from brokrest.plotting import Displayable, ViewPort

if typing.TYPE_CHECKING:
    from .rects import Box

__all__ = [
    "Topo",
    "TopoHandlerFunc",
    "TopoHandlerBase",
    "TopoHandlerProxy",
    "TopoInScope",
    "register_topo_handler",
    "enabled_topo_handlers",
]


class Topo(Displayable, abc.ABC):
    """
    A set of topologies.

    This is backed by `td.tensorclass`.
    If `.ndim == 0`, this is a single instance and you can call `.item()` on it.
    """

    @typing.final
    def __init__(self) -> None:
        if (batch_size := self._setup_shape()) is not NotImplemented:
            self.batch_size = batch_size

        self._checks()

        # This is done last for sure, since subclasses cannot have `__post_init__` defined.
        self._call_handlers()

    def _setup_shape(self) -> tuple[int, ...]:
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
        return auto_shape

    def _checks(self) -> None:
        return

    def _call_handlers(self):
        for handler in _TOPO_HANDLERS:
            handler(self)

    @typing.override
    def draw_on(self, vp: ViewPort) -> None:
        """
        Populate the canvas with `bokeh`, filter based on viewbox (`self.outer()`).
        """

        if self.plot is NotImplemented:
            return

        selected = self

        # Get the bounding box of `self`, and get rid of points not in the box.
        if (box := self.outer()) is not NotImplemented:
            visible_idx = box.visible(vp)
            selected = selected[visible_idx]

        selected.plot(vp.figure)

    @abc.abstractmethod
    def plot(self, figure: plotting.figure, /) -> None:
        """
        The implementation of `draw`.
        Subclass can choose to completely disable it by setting `plot = NotImplemented`.

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
        return torch.stack(values, dim=-1)

    @classmethod
    def from_dict(cls, items: cabc.Mapping[str, torch.Tensor]) -> typing.Self:
        broadcasted = _broadcast_tensor_dict(items)
        return cls(**broadcasted)


@typing.runtime_checkable
class TopoHandlerFunc(typing.Protocol):
    def __repr__(self) -> str: ...
    def __call__(self, topo: Topo, /) -> None: ...


class TopoHandlerBase(TopoHandlerFunc, abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, topo: Topo, /) -> None:
        raise NotImplementedError

    @ctxl.contextmanager
    def enable(self) -> cabc.Generator[typing.Self]:
        """
        Enable the handler in the scope under this context manager.
        """

        _TOPO_HANDLERS.append(self)
        try:
            yield self
        finally:
            _ = _TOPO_HANDLERS.pop()


@dcls.dataclass(frozen=True)
class TopoHandlerProxy(TopoHandlerBase):
    """
    The proxy object for handler, s.t. we can make priting easier,
    compared to using a `ctxl.contextmanager` closure directly.
    """

    handler: cabc.Callable[[Topo], None]
    "The handler that is wrapped."

    @typing.override
    def __repr__(self) -> str:
        return f"TopoHandler<{self.handler.__name__}>"

    @typing.override
    def __call__(self, topo: Topo, /):
        self.handler(topo)


def register_topo_handler(handler: TopoHandlerFunc, /):
    """
    Register a `TopoHandler` to call on `Topo` once initialization is done.
    """

    return TopoHandlerProxy(handler)


def enabled_topo_handlers():
    """
    Get all the currently enabled handlers (under the context managers) in a tuple.
    """

    return tuple(_TOPO_HANDLERS)


@dcls.dataclass(frozen=True)
class TopoInScope(TopoHandlerBase, Displayable):
    """
    Get all the topo in scope.
    """

    topo_list: list[Topo] = dcls.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.topo_list)

    def __getitem__(self, idx: int) -> Topo:
        return self.topo_list[idx]

    def __iter__(self) -> cabc.Generator[Topo, typing.Any, None]:
        yield from self.topo_list

    def __repr__(self) -> str:
        return repr(self.topo_list)

    def __call__(self, topo) -> None:
        self.topo_list.append(topo)

    @typing.override
    def draw_on(self, vp: ViewPort, /) -> None:
        for topo in self.topo_list:
            topo.draw_on(vp)


_TOPO_HANDLERS: list[TopoHandlerFunc] = []


def _broadcast_tensor_dict(
    items: cabc.Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Broadcast the tensors in a mapping from string to tensors to the same shape.
    """

    keys = list(items.keys())
    vals = [items[k] for k in keys]
    return {k: v for k, v in zip(keys, torch.broadcast_tensors(*vals))}
