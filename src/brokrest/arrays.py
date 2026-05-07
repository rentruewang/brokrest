# Copyright (c) The BrokRest Authors - All Rights Reserved

"The base classes for array."

import dataclasses as dcls
import functools
import operator
import types
import typing
from collections import abc as cabc

import numpy as np
from numpy import rec

__all__ = ["Array", "ArrayDict", "ArrayOrDict", "array_dict_dataclass"]

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


@typing.dataclass_transform(eq_default=False)
def array_dict_dataclass(cls: type[ArrayDict]):
    return dcls.dataclass(eq=False)(cls)


@array_dict_dataclass
class ArrayDict:

    def __post_init__(self):
        _ = self.shape

    def __array__(self, copy: bool = True) -> np.ndarray:
        values = [np.asarray(val, dtype=val.dtype, copy=copy) for val in self.values()]
        return rec.fromarrays(values, dtype=self.dtype)

    def __len__(self) -> int:
        return self.shape[0]

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

    def __add__(self, other):
        return self.__arithmetic(other, operator.add)

    def __sub__(self, other):
        return self.__arithmetic(other, operator.add)

    def __mul__(self, other):
        return self.__arithmetic(other, operator.mul)

    def __truediv__(self, other):
        return self.__arithmetic(other, operator.truediv)

    def __floordiv__(self, other):
        return self.__arithmetic(other, operator.floordiv)

    def __pow__(self, other):
        return self.__arithmetic(other, operator.pow)

    def __matmul__(self, other):
        return self.__arithmetic(other, operator.matmul)

    @typing.no_type_check
    def __eq__(self, other):
        return self.__arithmetic(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other):
        return self.__arithmetic(other, operator.ne)

    def __gt__(self, other):
        return self.__arithmetic(other, operator.gt)

    def __ge__(self, other):
        return self.__arithmetic(other, operator.ge)

    def __lt__(self, other):
        return self.__arithmetic(other, operator.lt)

    def __le__(self, other):
        return self.__arithmetic(other, operator.lt)

    def __arithmetic(self, other, op) -> typing.Self:
        if isinstance(self, ArrayDict) and isinstance(other, ArrayDict):
            if type(self) != type(other):
                return NotImplemented

            return type(self)(**{key: op(self[key], other[key]) for key in self.keys()})

        # Broadcast to every element if it's not decomposible (`ArrayDict`).
        return self.apply(lambda arr: op(arr, other))

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
            raise RuntimeError(
                f"ArrayDict.apply failed for {type(self)=}, {function=}."
            ) from e

    def swapaxes(self, axis0: int, axis1: int, /) -> typing.Self:
        return self.apply(lambda arr: arr.swapaxes(axis0, axis1))

    def transpose(self, axes: cabc.Sequence[int]) -> typing.Self:
        return self.apply(lambda arr: arr.transpose(axes))

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

    def squeeze(self, axis: int) -> typing.Self:
        if self.shape[axis] != 1:
            raise ValueError(
                "Cannot squeeze a dimension that is not 1! "
                f"{self.shape=}, {self.shape[axis]=}."
            )

        return self.reshape(*_shape_except_axis(self.shape, axis=axis))

    def flatten(self) -> typing.Self:
        return self.apply(lambda arr: arr.flatten())

    @classmethod
    def stack(cls, insts: cabc.Sequence[typing.Self], /, axis: int = 0) -> typing.Self:
        if not all(isinstance(inst, cls) for inst in insts):
            raise TypeError(f"Not all instances are subclasses of {cls}")

        array = np.stack([np.asarray(inst) for inst in insts], axis=axis)
        keys = insts[0].keys()
        return cls(**{key: array[key] for key in keys})

    @classmethod
    def concat(cls, insts: cabc.Sequence[typing.Self], /, axis: int = 0) -> typing.Self:
        if not all(isinstance(inst, cls) for inst in insts):
            raise TypeError(f"Not all instances are subclasses of {cls}")

        array = np.concat([np.asarray(inst) for inst in insts], axis=axis)
        keys = insts[0].keys()
        return cls(**{key: array[key] for key in keys})


def _shape_except_axis(shape: tuple[int, ...], axis: int):
    axis %= len(shape)
    return *shape[:axis], *shape[axis + 1 :]
