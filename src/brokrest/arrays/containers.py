# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
from numpy import rec

from brokrest.typing import IntArray

from .arrays import Array, ArrayOrDict, array_dataclass

__all__ = ["ArrayDict", "ArrayList"]


@array_dataclass
class ArrayDict(Array):
    def __post_init__(self):
        _ = self.shape

        for key, field in self.items():
            if not isinstance(field, np.generic | np.ndarray | Array):
                raise ValueError(
                    f"Field {field} at {key} is not a numpy value or an `ArrayDict`."
                )

    @typing.override
    def __array__(self, copy: bool = True) -> np.ndarray:
        values = [np.asarray(val, dtype=val.dtype, copy=copy) for val in self.values()]
        return rec.fromarrays(values, dtype=self.dtype)

    def keys(self):
        return self.fields().keys()

    def values(self):
        return self.fields().values()

    def items(self):
        return self.fields().items()

    @typing.override
    def _ufunc_2(self, other, op) -> typing.Self:
        if isinstance(self, ArrayDict) and isinstance(other, ArrayDict):
            if type(self) != type(other):
                return NotImplemented

            return type(self)(**{key: op(self[key], other[key]) for key in self.keys()})

        # Broadcast to every element if it's not decomposible (`ArrayDict`).
        return self.apply(lambda arr: op(arr, other))

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

    @typing.override
    def apply(self, function: cabc.Callable[..., typing.Any]) -> typing.Self:
        try:
            return type(self)(**{k: function(v) for k, v in self.items()})
        except Exception as e:
            raise RuntimeError(
                f"ArrayDict.apply failed for {type(self)=}, {function=}."
            ) from e

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
        return {name: getattr(self, name) for name in self._field_names()}

    @property
    @typing.override
    def dtype(self) -> np.dtype:
        return self.__dtype_cached

    @functools.cached_property
    def __dtype_cached(self) -> np.dtype:
        return np.dtype([(key, val.dtype) for key, val in self.items()])

    @classmethod
    @typing.override
    def from_array(cls, array: np.ndarray) -> typing.Self:
        return cls(**{key: array[key] for key in cls._field_names()})

    @classmethod
    def _field_names(cls):
        return [field.name for field in dcls.fields(cls)]


@array_dataclass
class ArrayList[T: Array = Array]:
    data: T
    """
    The data stored (vectorized). Each item can be of different length (jagged).
    """

    split: IntArray
