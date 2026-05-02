# Copyright (c) The BrokRest Authors - All Rights Reserved

import typing

import numpy as np
from numpy import typing as npt

__all__ = ["IntArray", "FloatArray", "DoubleArray"]

IntArray: typing.TypeAlias = npt.NDArray[np.int_]
FloatArray: typing.TypeAlias = npt.NDArray[np.float32]
DoubleArray: typing.TypeAlias = npt.NDArray[np.float64]
