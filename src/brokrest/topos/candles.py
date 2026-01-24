# Copyright (c) The BrokRest Authors - All Rights Reserved

"A collection of topologies that are geometric shapes."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from typing import ClassVar, Self

import torch
from bokeh.plotting import figure as Figure
from pandas import DataFrame
from torch import Tensor

from .rects import Box
from .topos import Shape

__all__ = ["Candle", "CandleLooks", "BothCandle", "LeftCandle"]


@dcls.dataclass
class CandleLooks:
    "The appearances for candles."

    up_line: str = "green"
    "Color for the boundary of up."

    up_fill: str = "white"
    "The color for interior of up."

    down_line: str = "red"
    "Color for the boundary of down."

    down_fill: str = "red"
    "The color for interior of down."

    line_width: int = 2
    "How wide should the surrounding line be?"

    width_ratio: float = 0.7
    "How wide should each bar be? Max 1 min 0"

    def __post_init__(self) -> None:
        if not 0 <= self.width_ratio <= 1:
            raise ValueError(f"{self.width_ratio=} is not in range [0, 1]")

    def invert(self) -> Self:
        return dcls.replace(
            self,
            down_line=self.up_line,
            up_line=self.down_line,
            up_fill=self.down_fill,
            down_fill=self.up_fill,
        )


@dcls.dataclass
class Candle(Shape, ABC):
    """
    A candle on the candle chart
    """

    KEYS: ClassVar[tuple[str, ...]] = "enter", "exit", "low", "high"

    looks: CandleLooks = dcls.field(default_factory=CandleLooks)
    """
    The profile of the candles' appearances.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        if torch.any(self.low > self.high):
            raise ValueError(
                f"Min value: {self.low} must be smaller than max value: {self.high}."
            )

        self._check_value_in_range(self.enter, "entering")
        self._check_value_in_range(self.exit, "exiting")

    def _check_value_in_range(self, value: Tensor, desc: str) -> None:
        "Check if the given value is in the range [min, max]."

        if torch.any(self.low > value) or torch.any(self.high < value):
            message = " ".join(
                [
                    f"{desc.capitalize()} value: {self.enter},",
                    f"but min={self.low}, max={self.high}.",
                ]
            )
            raise ValueError(message)

    @property
    def enter(self) -> Tensor:
        """
        The entering position of this candle.
        """

        return self["enter"]

    @property
    def exit(self) -> Tensor:
        """
        The exiting position of this candle.
        """

        return self["exit"]

    @property
    def low(self) -> Tensor:
        """
        The minimum value of the candle.
        """

        return self["low"]

    @property
    def high(self) -> Tensor:
        """
        The maximum value of the candle.
        """

        return self["high"]

    @property
    @abc.abstractmethod
    def center(self) -> Tensor:
        "The time at which this candle occurs."

        ...

    @property
    @abc.abstractmethod
    def max_width(self) -> float:
        "The maximum width this candle can occupy."

        ...

    @property
    @abc.abstractmethod
    def left(self) -> Tensor:
        "The left side of the candle."

        ...

    @property
    @abc.abstractmethod
    def right(self) -> Tensor:
        "The right side of the candle."

        ...

    @property
    def inc(self) -> Tensor:
        "Is increasing."
        return (self.exit - self.enter) >= 0

    @property
    def dec(self) -> Tensor:
        "Is decreasing."
        return (self.exit - self.enter) < 0

    @typing.override
    def _draw(self, figure: Figure):
        # The center bars for the candles.
        _ = figure.segment(
            x0=self.center.numpy(),
            y0=self.high.numpy(),
            x1=self.center.numpy(),
            y1=self.low.numpy(),
            color="black",
        )

        width = self.max_width * 0.7

        # The body of candles that are decreasing.
        _ = figure.vbar(
            x=self.center[self.dec].numpy(),
            width=width,
            top=self.enter[self.dec].numpy(),
            bottom=self.exit[self.dec].numpy(),
            fill_color=self.looks.down_fill,
            color=self.looks.down_line,
            line_width=self.looks.line_width,
        )

        # The body of candles that are increasing.
        _ = figure.vbar(
            x=self.center[self.inc].numpy(),
            width=width,
            top=self.enter[self.inc].numpy(),
            bottom=self.exit[self.inc].numpy(),
            fill_color=self.looks.up_fill,
            line_color=self.looks.up_line,
            line_width=self.looks.line_width,
        )

    @typing.override
    @abc.abstractmethod
    def ordering(self) -> Tensor:
        """
        As the candles are organized by time, ordering must be present.
        """


class BothCandle(Candle):
    """
    A candle that has a left side and a right side.
    """

    KEYS: ClassVar[tuple[str, ...]] = *Candle.KEYS, "start", "end"

    def __post_init__(self) -> None:
        super().__post_init__()

        if torch.any(self.start > self.end):
            raise ValueError(
                f"Starting time {self.start} must be before than ending time: {self.end}."
            )

    def continuous(self) -> bool:
        nexts = self[1:]
        prevs = self[:-1]
        return torch.allclose(nexts.start, prevs.end)

    @property
    def start(self):
        return self["start"]

    @property
    def end(self):
        return self["end"]

    @property
    @typing.override
    def center(self) -> Tensor:
        "The centered times."
        return (self.end + self.start) / 2

    @property
    @typing.override
    def max_width(self) -> float:
        "The width for each candle."
        return min(self.end - self.start).item()

    @property
    @typing.override
    def left(self):
        return self.center - self.max_width

    @property
    @typing.override
    def right(self):
        return self.center + self.max_width

    @typing.override
    def _outer(self):
        return Box.init(x_0=self.start, x_1=self.end, y_0=self.low, y_1=self.high)

    @typing.override
    def ordering(self) -> Tensor:
        return self.start


class LeftCandle(Candle):
    """
    The candle that only has the starting time defined (timing is implicit).
    """

    KEYS: ClassVar[tuple[str, ...]] = *Candle.KEYS, "start"

    @property
    def start(self):
        return self["start"]

    @property
    @typing.override
    def center(self) -> Tensor:
        "The centered times."
        return self.start + self.max_width / 2

    @property
    @typing.override
    def max_width(self) -> float:
        "The width for each candle. This is the minimal interval between starts."

        return min(self.start[1:] - self.start[:-1]).item()

    @property
    @typing.override
    def left(self):
        return self.start

    @property
    @typing.override
    def right(self):
        return self.end

    @property
    def end(self):
        return self.start + self.max_width

    @typing.override
    def _outer(self):
        return Box.init(x_0=self.start, x_1=self.end, y_0=self.low, y_1=self.high)

    @typing.override
    def ordering(self) -> Tensor:
        return self.start


def dataframe_factory(df: DataFrame, /) -> Candle:
    """
    Parse an input dataframe into corresponding ``Candle``.

    Args:
        df:
            A dataframe.
            Must have one of the combinations of keys:
            - 'start', 'end', 'enter', 'exit', 'low','high'
            - 'start', 'enter', 'exit', 'low', 'high'


    Returns:
        A ``Candle`` instance, type dependent on the input keys.
    """

    if (
        inst := _try_init_with_type_and_keys(
            df, BothCandle, "start", "end", "enter", "exit", "low", "high"
        )
    ) is not None:
        return inst

    if (
        inst := _try_init_with_type_and_keys(
            df, LeftCandle, "start", "enter", "exit", "low", "high"
        )
    ) is not None:
        return inst

    raise NotImplementedError(
        f"Do not recognize the key combination from df: {list(df.columns)}"
    )


def _try_init_with_type_and_keys(df: DataFrame, typ: type[Candle], *keys: str):
    if not all(k in df.columns for k in keys):
        return None

    dicts = {key: torch.tensor(df[key].tolist()) for key in keys}
    return typ.init(**dicts)
