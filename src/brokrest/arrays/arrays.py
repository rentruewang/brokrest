# Copyright (c) The BrokRest Authors - All Rights Reserved

"The base classes for array."

import abc
import dataclasses as dcls
import operator
import types
import typing
from collections import abc as cabc

import numpy as np

__all__ = ["Array", "ArrayOrDict", "array_dataclass"]

type ArrayOrDict = Array | np.generic | np.ndarray

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
def array_dataclass(cls: type[Array]):
    return dcls.dataclass(eq=False)(cls)


@array_dataclass
class Array(abc.ABC):
    def __post_init__(self):
        _ = self.shape

        for key, field in self.items():
            if not isinstance(field, np.generic | np.ndarray | Array):
                raise ValueError(
                    f"Field {field} at {key} is not a numpy value or an `ArrayDict`."
                )

    @abc.abstractmethod
    def __array__(self, copy: bool = True) -> np.ndarray:
        raise NotImplementedError

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
        return self._ufunc_2(other, operator.add)

    def __sub__(self, other):
        return self._ufunc_2(other, operator.add)

    def __mul__(self, other):
        return self._ufunc_2(other, operator.mul)

    def __truediv__(self, other):
        return self._ufunc_2(other, operator.truediv)

    def __floordiv__(self, other):
        return self._ufunc_2(other, operator.floordiv)

    def __pow__(self, other):
        return self._ufunc_2(other, operator.pow)

    def __matmul__(self, other):
        return self._ufunc_2(other, operator.matmul)

    @typing.no_type_check
    def __eq__(self, other):
        return self._ufunc_2(other, operator.eq)

    @typing.no_type_check
    def __ne__(self, other):
        return self._ufunc_2(other, operator.ne)

    def __gt__(self, other):
        return self._ufunc_2(other, operator.gt)

    def __ge__(self, other):
        return self._ufunc_2(other, operator.ge)

    def __lt__(self, other):
        return self._ufunc_2(other, operator.lt)

    def __le__(self, other):
        return self._ufunc_2(other, operator.lt)

    @abc.abstractmethod
    def _ufunc_2(self, other, op) -> typing.Self:
        "Binary ufunc."
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, function: cabc.Callable[..., typing.Any]) -> typing.Self:
        "Apply transformation elementwise."

        raise NotImplementedError

    def keys(self):
        return self.fields().keys()

    def values(self):
        return self.fields().values()

    def items(self):
        return self.fields().items()

    @abc.abstractmethod
    def fields(self) -> dict[str, ArrayOrDict]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        return np.multiply.reduce(self.shape).item()

    @property
    def ndim(self) -> int:
        "The number of dimensions in this current `ArrayDict`."

        return len(self.shape)

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        "The shape of the `ArrayDict`."

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

    def swapaxes(self, axis0: int, axis1: int, /) -> typing.Self:
        return self.apply(lambda arr: arr.swapaxes(axis0, axis1))

    def transpose(self, axes: cabc.Sequence[int]) -> typing.Self:
        return self.apply(lambda arr: arr.transpose(axes))

    def reshape(self, *dims: int) -> typing.Self:
        return self.apply(lambda v: v.reshape(*dims))

    @classmethod
    @abc.abstractmethod
    def from_array(cls, array: np.ndarray) -> typing.Self:
        raise NotImplementedError

    @classmethod
    def stack(cls, insts: cabc.Sequence[typing.Self], /, axis: int = 0) -> typing.Self:
        if not all(isinstance(inst, cls) for inst in insts):
            raise TypeError(f"Not all instances are subclasses of {cls}")

        array = np.stack([np.asarray(inst) for inst in insts], axis=axis)
        return cls.from_array(array)

    @classmethod
    def concat(cls, insts: cabc.Sequence[typing.Self], /, axis: int = 0) -> typing.Self:
        if not all(isinstance(inst, cls) for inst in insts):
            raise TypeError(f"Not all instances are subclasses of {cls}")

        array = np.concat([np.asarray(inst) for inst in insts], axis=axis)
        return cls.from_array(array)


def _shape_except_axis(shape: tuple[int, ...], axis: int):
    axis %= len(shape)
    return *shape[:axis], *shape[axis + 1 :]
