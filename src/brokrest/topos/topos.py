# Copyright (c) The BrokRest Authors - All Rights Reserved

"Topologies API `Topo`, representing a set of shapes."

import abc
import contextlib as ctxl
import copy
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
import pandas as pd
from bokeh import plotting
from numpy import typing as npt

from brokrest.plotting import Displayable, ViewPort
from brokrest.typing import FloatArray, IntArray

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


@dcls.dataclass
class Topo(Displayable, abc.ABC):
    """
    A set of topologies.

    This is a dataclass of arrays.
    If `.ndim == 0`, this is a single instance and you can call `item()` on it.
    """

    @typing.final
    def __post_init__(self) -> None:
        print("Before setup shape")
        self._shape = self._setup_shape()
        print("After setup shape", self.shape)
        self._sort_by_value()
        self._checks()

        # This is done last for sure, since subclasses cannot have `__post_init__` defined.
        self._call_handlers()

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @typing.overload
    def __getitem__(self, idx: str) -> FloatArray: ...
    @typing.overload
    def __getitem__(self, idx) -> typing.Self: ...
    @typing.no_type_check
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.array_dict()[idx]

        else:
            return self.apply(lambda arr: arr[idx])

    def _checks(self) -> None:
        return

    def _setup_shape(self) -> tuple[int, ...]:
        # If mapping is empty, the shape is `()`.
        if not (mapping := self.array_dict()):
            return ()

        shape, *rest = (arr.shape for arr in mapping.values())

        for other in rest:
            shape = _shape_prefix(shape, other)

        return shape

    def _call_handlers(self):
        for handler in _TOPO_HANDLERS:
            handler(self)

    @typing.no_type_check
    def _sort_by_value(self) -> None:
        "Sort according to `ordering`."

        if (ordering := self.ordering()) is NotImplemented:
            return

        # This is OK. Both are scalars.
        if self.ndim == 0:
            if ordering.size == 1:
                return

            raise ValueError(f"Ndim == 0, but {ordering=} has more than 1 index.")

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

        for key in self.keys():
            setattr(self, key, getattr(self, key)[ordering])

    def ordering(self) -> IntArray:
        """
        Return the argsort of the current `Topo`.

        If the collection doesn't need to be ordered, return `NotImplemented`.

        Returns:
            A 1D `npt.NDArray` of shape [len(self)],
            whose elements are permutation of `range(len(self))`,
            or `NotImplemented` if ordering doesn't exist.
        """

        return NotImplemented

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

    def apply(self, ufunc: "UFunc") -> typing.Self:
        self = copy.deepcopy(self)
        for key, val in self.array_dict().items():
            mapped = ufunc(val)
            setattr(self, key, mapped)
        self.__post_init__()
        return self

    def keys(self):
        return self.array_dict().keys()

    def values(self):
        return self.array_dict().values()

    def items(self):
        return self.array_dict().items()

    def array_dict(self) -> dict[str, npt.NDArray]:
        return dict(self._array_dict())

    def _array_dict(self):
        for key in self.__all_keys:
            item = getattr(self, key)

            if _is_numpy_type(item):
                yield key, item

    @functools.cached_property
    def __all_keys(self):
        return {key.name for key in dcls.fields(self)}

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self, *dims: int) -> typing.Self:
        return self.apply(lambda x: x.reshape(*dims))

    def flatten(self) -> typing.Self:
        return self.reshape(-1)

    @classmethod
    def from_dict(cls, items: cabc.Mapping[str, FloatArray]) -> typing.Self:
        broadcasted = _broadcast_tensor_dict(items)
        return cls(**broadcasted)

    @staticmethod
    def stack[T: Topo](items: list[T], /, axis: int = 0) -> T:
        return _stack_or_cat(
            items,
            merge_array=np.stack,
            name="stack",
            axis=axis,
        )

    @staticmethod
    def cat[T: Topo](items: list[T], /, axis: int = 0) -> T:
        return _stack_or_cat(
            items,
            merge_array=np.concatenate,
            name="cat",
            axis=axis,
        )


class UFunc(typing.Protocol):
    def __call__(self, arr) -> typing.Any: ...


def _is_numpy_type(x):
    return isinstance(x, (np.ndarray, np.generic))


class MergeArray(typing.Protocol):
    def __call__(self, arr, /, axis: int = 0) -> typing.Any: ...


@typing.no_type_check
def _stack_or_cat[T: Topo](
    items: list[T],
    merge_array: MergeArray,
    name: str,
    axis: int,
) -> T:
    if not len(items):
        raise ValueError("Empty list encountered.")

    if len(types := {type(item) for item in items}) != 1:
        raise TypeError(
            f"{name.capitalize()} only supported for topos of the same type. Got {types}."
        )

    target_type = list(types)[0]
    dicts = pd.DataFrame([item.array_dict() for item in items])

    result = {}
    for col in dicts.columns:
        result[col] = merge_array(dicts[col], axis=axis)

    return target_type(**result)


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


def _shape_prefix(shape: tuple[int, ...], other: tuple[int, ...]) -> tuple[int, ...]:
    "Find the common prefix between `shape` and `other`."

    result: list[int] = []

    for s, o in zip(shape, other):
        if s != o:
            return tuple(result)

        result.append(s)

    return shape


def _broadcast_tensor_dict(
    items: cabc.Mapping[str, npt.NDArray],
) -> dict[str, npt.NDArray]:
    """
    Broadcast the tensors in a mapping from string to tensors to the same shape.
    """

    keys = list(items.keys())
    vals = [items[k] for k in keys]
    return {k: v for k, v in zip(keys, np.broadcast_arrays(*vals))}
