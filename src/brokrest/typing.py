# Copyright (c) The BrokRest Authors - All Rights Reserved

import numpy as np
from numpy import typing as npt

__all__ = ["FloatArray", "IntArray", "BoolArray", "UIntArray"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool]
UIntArray = npt.NDArray[np.uint64]
ArrayOrScalar = FloatArray | IntArray | BoolArray | UIntArray | bool | int | float
