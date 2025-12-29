# Copyright (c) The BrokRest Authors - All Rights Reserved

"The candles and candle bar charts."

import dataclasses as dcls
from collections.abc import Sequence
from typing import NamedTuple, Self

from .painters import Box, Canvas
from .vectors import Vec2d


@dcls.dataclass(frozen=True)
class Candle:
    """
    One single candle bar.
    """

    enter: float
    """
    The entering position of this candle.
    """

    exit: float
    """
    The exiting position of this candle.
    """

    min: float
    """
    The minimum value of the candle.
    """

    max: float
    """
    The maximum value of the candle.
    """

    start: int
    """
    The starting time of the candle.
    """

    end: int
    """
    The ending time of the candle.
    """

    def __post_init__(self) -> None:
        mini, maxi = self.min, self.max

        if maxi < mini:
            raise ValueError(
                f"Min value: {mini} should be smaller than max value: {maxi}."
            )

        if not mini <= self.enter <= maxi:
            raise ValueError(f"Start value: {self.enter}, but min={mini}, max={maxi}.")

        if not mini <= self.exit <= maxi:
            raise ValueError(f"End value: {self.exit}, but min={mini}, max={maxi}.")

    def __union__(self, rhs: Self):
        return self.merge(rhs)

    def merge(self, rhs: Self, /) -> Self:
        """
        Merging 2 candle bars.
        Assuming ``self`` to be the left candle and ``rhs`` to be the right candle.

        Args:
            rhs: Another ``Candle`` that will be connected to the right.

        Returns:
            A new ``Candle``.
        """

        return type(self)(
            enter=self.enter,
            exit=rhs.exit,
            min=min(self.min, rhs.min),
            max=max(self.max, rhs.max),
            start=min(self.start, rhs.start),
            end=min(self.end, rhs.end),
        )

    def overlaps(self, rhs: Self, /) -> bool:
        """
        Whether or not ``self`` and ``rhs`` overlaps in time.

        Args:
            rhs: Another ``Candle``.

        Returns:
            A boolean.
        """

        def tuples_to_check():
            # ``self`` contains ``rhs``.
            yield _FourTuple(self.start, rhs.start, rhs.end, self.end)

            # ``rhs`` contains ``self``.
            yield _FourTuple(rhs.start, self.start, self.end, rhs.end)

            # ``self`` left side overlaps ``rhs`` right side.
            yield _FourTuple(rhs.start, self.start, rhs.end, self.end)

            # ``self`` right side overlaps ``rhs`` left side.
            yield _FourTuple(self.start, rhs.start, self.end, rhs.end)

        return any(t.is_sorted() for t in tuples_to_check())

    def plot(self, canvas: Canvas) -> None:
        "Plot method for ``Candle``."

        color = "green" if self.exit - self.enter > 0 else "red"

        canvas.fill(
            box=Box(
                left=self.start,
                right=self.end,
                top=max(self.enter, self.exit),
                bottom=min(self.enter, self.exit),
            ),
            color=color,
        )

        middle = (self.end + self.start) / 2
        canvas.line(
            Vec2d(middle, self.min),
            Vec2d(middle, self.max),
            color=color,
        )


@dcls.dataclass(frozen=True)
class CandleChart:
    """
    ``CandleChart`` is a view into the underlying ``Dataset``.
    """

    values: Sequence[float]
    """
    The values at each time frame for the candle chart.
    This contains the boundaries, so its ``len`` == ``len(self) + 1``.
    """

    start: int
    """
    The start of the viewport.
    """

    end: int
    """
    The end of the viewport.
    """

    interval: int
    """
    The number of candles to generate.
    """

    def __post_init__(self) -> None:
        if (self.end - 1 - self.start) % self.interval != 0:
            raise NotImplementedError(
                f"Interval = {self.interval} should fully divide the range {self.start}-{self.end}."
            )

        if not all(isinstance(v, float) for v in self.values):
            raise ValueError(f"Values should be a list of floats. Got {self.values}.")

    def __len__(self) -> int:
        return (self.end - 1 - self.start) // self.interval

    def __getitem__(self, idx: int) -> Candle:
        start = self.start + idx * self.interval
        end = start + self.interval

        return Candle(
            enter=self.values[start],
            exit=self.values[end],
            min=min(self.values[start : end + 1]),
            max=max(self.values[start : end + 1]),
            start=start,
            end=end,
        )

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def plot(self, canvas: Canvas) -> None:
        "Plot method for ``CandleChart``."

        for candle in self:
            candle.plot(canvas)

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Self:
        """
        Create an instance of ``CandleChart`` from values.

        Args:
            values: The underlying values.

        Returns:
            A candle chart.
        """
        return cls(
            values=values,
            start=0,
            end=len(values),
            interval=1,
        )


class _FourTuple(NamedTuple):
    a: float
    b: float
    c: float
    d: float

    def is_sorted(self) -> bool:
        "Check if the tuple is in the [a, b, c, d] order."

        return self.a <= self.b <= self.c <= self.d
