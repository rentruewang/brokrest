# Copyright (c) The BrokRest Authors - All Rights Reserved

"The base classes for array."

import dataclasses as dcls
import functools
import types
import typing
from collections import abc as cabc

import numpy as np

__all__ = ["Array", "ArrayDict", "ArrayOrDict", "JaggedArray"]

type Array = np.ndarray | np.generic
type ArrayOrDict = Array | ArrayDict

ArrayDictRowIdx: typing.TypeAlias = (
    None
    | types.EllipsisType
    | int
    | slice
    | list[int]
    | np.ndarray
    | tuple[typing.Any, ...]
)
"The row wise index access for `ArrayDict`."


@dcls.dataclass
class ArrayDict:

    def __post_init__(self):
        _ = self.shape

    def __array__(self) -> np.ndarray:
        values = [np.asarray(val, dtype=val.dtype) for val in self.values()]
        expand = [np.expand_dims(val, axis=0) for val in values]
        return np.concat(expand, axis=0)

    def __len__(self) -> int:
        return len(self.shape)

    @typing.overload
    def __getitem__(self, key: str, /) -> ArrayOrDict: ...

    @typing.overload
    def __getitem__(self, key: ArrayDictRowIdx, /) -> typing.Self: ...

    @typing.no_type_check
    def __getitem__(self, key, /):
        if isinstance(key, str):
            return self.fields()[key]

        else:
            return self.apply(lambda v: v[key])

    def __iter__(self) -> cabc.Iterator[typing.Self]:
        for i in range(len(self)):
            yield self[i]

    def keys(self):
        return self.fields().keys()

    def values(self):
        return self.fields().values()

    def items(self):
        return self.fields().items()

    @property
    def shape(self) -> tuple[int, ...]:
        "The shape of the `ArrayDict`."
        return self._shape

    @functools.cached_property
    def _shape(self):
        values = list(self.values())
        shapes = {val.shape for val in values}

        if len(shapes) != 1:
            raise ValueError(f"Multiple shapes found in {type(self)}: {shapes}.")

        return list(shapes)[0]

    def apply(self, function: cabc.Callable[..., typing.Any]) -> typing.Self:
        try:
            return type(self)(**{k: function(v) for k, v in self.items()})
        except Exception as e:
            raise type(e)(f"ArrayDict.apply failed for {self=}, {function=}.") from e

    def transpose(self, axes: cabc.Sequence[int]) -> typing.Self:
        return self.apply(lambda arr: np.transpose(arr, axes))

    def reshape(self, *dims: int) -> typing.Self:
        return self.apply(lambda v: v.reshape(*dims))

    @property
    def size(self) -> int:
        return np.multiply.reduce(self.shape).item()

    @property
    def ndim(self) -> int:
        "The number of dimensions in this current `ArrayDict`."

        return len(self.shape)

    def fields(self) -> dict[str, ArrayOrDict]:
        """
        The fields in the array.

        Since python 3.7, the dict values are ordered,
        so we can reliably use dict.values() and stack it.
        """

        fields = self.__fields_cached

        for key, arr in fields.items():
            if not isinstance(arr, np.ndarray | np.generic | ArrayDict):
                raise TypeError(
                    f"The field at {key=} has value {arr=}, which is not a numpy array."
                )

        return fields

    @functools.cached_property
    def __fields_cached(self) -> dict[str, ArrayOrDict]:
        fields = dcls.fields(self)
        names = [field.name for field in fields]
        return {name: getattr(self, name) for name in names}

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype_cached

    @functools.cached_property
    def __dtype_cached(self) -> np.dtype:
        return np.dtype([(key, val.dtype) for key, val in self.items()])

    def item(self) -> typing.Self:
        if self.size != 1:
            raise RuntimeError(
                f"Cannot call .item() on array with multiple elements: {self.shape=}."
            )

        return self.apply(lambda arr: arr.reshape(1))[0]
