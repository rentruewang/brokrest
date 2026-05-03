# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
import typing

from brokrest.typing import FloatArray

__all__ = ["Importance", "Window"]


class Importance(typing.Protocol):
    """
    A 1D probabilitiy distribution that works in batches.

    It computes the input distance values, whose +- signs show above / below the lines),
    and re-sample their importances.

    Outputs the original distance matrix filtered.
    """

    def __call__(self, dists: FloatArray, /) -> FloatArray: ...


@dcls.dataclass(frozen=True)
class Window:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"Invalid configuration: {self.lower=}, {self.upper=}.")

    def __call__(self, dists: FloatArray, /) -> FloatArray:
        in_range = (dists >= self.lower) & (dists <= self.upper)
        return dists * in_range
