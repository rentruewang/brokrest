# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta

import numpy as np
from numpy.typing import NDArray
from shapely import MultiPoint, Point

from brokrest.errors import BrokrestError


@dcls.dataclass(frozen=True)
class Ticks:
    start: DateTime
    end: DateTime
    interval: TimeDelta

    def __len__(self) -> int:
        return (self.end - self.start) // self.interval

    def __getitem__(self, idx: int) -> DateTime:
        return self.start + idx * self.interval

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __array__(self) -> NDArray:
        return np.arange(
            start=self.start.timestamp(),
            stop=self.end.timestamp(),
            step=self.interval.total_seconds(),
        )


@dcls.dataclass(frozen=True)
class PriceHistory:
    values: NDArray
    ticks: Ticks

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise_input_malformed(self)

        if len(self.values) != len(self.ticks):
            raise_input_malformed(self)

    def points(self) -> MultiPoint:
        return MultiPoint(
            [Point(t.timestamp(), v) for t, v in zip(self.ticks, self.values)]
        )


def raise_input_malformed(hist: PriceHistory, /):
    raise PriceValueError(
        "Prices or ticks must be 1 dimensional, and of equal size. "
        f"Got {hist.values.shape=} and {hist.ticks=}"
    )


class PriceValueError(BrokrestError, ValueError): ...
