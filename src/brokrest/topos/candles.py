# Copyright (c) The BrokRest Authors - All Rights Reserved

"A collection of topologies that are geometric shapes."

import abc
import dataclasses as dcls
import functools
import typing

import numpy as np
import pandas as pd
import shapely
import tensordict as td
import torch
from bokeh import plotting

from ._turnaround import simple_keep_turnaround_segments
from .lines import Point
from .polygons import Polygon
from .rects import Box
from .topos import Topo

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

    def invert(self) -> typing.Self:
        return dcls.replace(
            self,
            down_line=self.up_line,
            up_line=self.down_line,
            up_fill=self.down_fill,
            down_fill=self.up_fill,
        )


class Candle(Topo, abc.ABC):
    """
    A candle on the candle chart
    """

    enter: np.ndarray
    "The entering position of this candle."

    exit: np.ndarray
    "The exiting position of this candle."

    low: np.ndarray
    "The minimum value of the candle."

    high: np.ndarray
    "The maximum value of the candle."

    @typing.override
    def _checks(self) -> None:
        if torch.any(self.low > self.high):
            raise ValueError(
                f"Min value: {self.low} must be smaller than max value: {self.high}."
            )

        self._check_value_in_range(self.enter, "entering")
        self._check_value_in_range(self.exit, "exiting")

    def _check_value_in_range(self, value: np.ndarray, desc: str) -> None:
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
    def center_y(self):
        return self.low / 2 + self.high / 2

    @property
    @abc.abstractmethod
    def center(self) -> np.ndarray:
        "The time at which this candle occurs."

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_width(self) -> float:
        "The maximum width this candle can occupy."

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def left(self) -> np.ndarray:
        "The left side of the candle."

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def right(self) -> np.ndarray:
        "The right side of the candle."

        raise NotImplementedError

    @property
    def inc(self) -> np.ndarray:
        "Is increasing."

        return self.direction >= 0

    @property
    def dec(self) -> np.ndarray:
        "Is decreasing."

        return self.direction < 0

    @property
    def direction(self):
        "The direction for each candle. 1 for up and -1 for down."

        return (self.exit - self.enter).sign()

    @typing.no_type_check
    def center_points(self, enter_exit: bool = True) -> Point:
        coords = self.top_bottom_bounds(enter_exit=enter_exit)
        return coords[..., 0] / 2 + coords[..., 1] / 2

    def to_turnaround_segments(self):
        """
        Convert to the segments you commonly see in a stock;
        where each folding point of the segment represents a bounce (up to down or down to up).

        Switch to a vectorized implementation if possible.
        """

        return simple_keep_turnaround_segments(self)

    def top_bottom_bounds(self, enter_exit: bool = True) -> Point:
        """
        The top and bottom boundary points, stacked in the last dimension.
        """

        if enter_exit:
            top = np.where(self.inc, self.exit, self.enter)
            bottom = np.where(self.dec, self.exit, self.enter)
        else:
            top = self.high
            bottom = self.low

        assert (top >= bottom).all()

        top_coords = Point(x=self.center, y=top)
        bottom_coords = Point(x=self.center, y=bottom)

        return td.stack([top_coords, bottom_coords], dim=-1)

    @typing.no_type_check
    def convex(self, enter_exit: bool = True):
        coords = self.top_bottom_bounds(enter_exit=enter_exit)
        point_set = shapely.MultiPoint(
            coords.transpose(-1, 0).flatten().tensor().numpy()
        )
        if not isinstance(cvx := point_set.convex_hull, shapely.Polygon):
            raise RuntimeError("Did not return a polygon.")

        return Polygon.from_shapely_polygon(cvx)

    @typing.override
    def plot(self, figure: plotting.figure) -> None:
        # The center bars for the candles.
        _ = figure.segment(
            x0=self.center, y0=self.high, x1=self.center, y1=self.low, color="black"
        )

        width = self.max_width * 0.7

        # The body of candles that are decreasing.
        _ = figure.vbar(
            x=self.center[self.dec],
            width=width,
            top=self.enter[self.dec],
            bottom=self.exit[self.dec],
            fill_color=self.looks.down_fill,
            color=self.looks.down_line,
            line_width=self.looks.line_width,
        )

        # The body of candles that are increasing.
        _ = figure.vbar(
            x=self.center[self.inc],
            width=width,
            top=self.enter[self.inc],
            bottom=self.exit[self.inc],
            fill_color=self.looks.up_fill,
            line_color=self.looks.up_line,
            line_width=self.looks.line_width,
        )

    @functools.cached_property
    def looks(self) -> CandleLooks:
        """
        The profile of the candles' appearances.
        """

        return CandleLooks()

    def where(self, after: float, before: float):
        """
        Select the candles in the range `[after, before]`.

        Args:
            after: Select the candles after this time.
            before: Select the candles before this time.

        Raises:
            ValueError: If `before < after`.
        """

        if before < after:
            raise ValueError(
                f"Where selects candles in the range [{after}, {before}], but {before} < {after}."
            )

        selected = (self.center >= before) & (self.center <= after)
        return self[selected]


class BothCandle(Candle):
    """
    A candle that has a left side and a right side.
    """

    start: np.ndarray
    "The starting time of the candle."

    end: np.ndarray
    "The ending time of the candle."

    @typing.override
    def _checks(self) -> None:
        super()._checks()

        if torch.any(self.start > self.end):
            raise ValueError(
                f"Starting time {self.start} must be before than ending time: {self.end}."
            )

    def continuous(self) -> bool:
        nexts = self[1:]
        prevs = self[:-1]
        return torch.allclose(nexts.start, prevs.end)

    @property
    @typing.override
    def center(self) -> np.ndarray:
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
        return Box(x_0=self.start, x_1=self.end, y_0=self.low, y_1=self.high)


class LeftCandle(Candle):
    """
    The candle that only has the starting time defined (timing is implicit).
    """

    start: np.ndarray
    "The starting time of the candle."

    @property
    @typing.override
    def center(self) -> np.ndarray:
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
        return Box(x_0=self.start, x_1=self.end, y_0=self.low, y_1=self.high)


def dataframe_to_candles(df: pd.DataFrame, /) -> Candle:
    """
    Parse an input dataframe into corresponding `Candle`.

    Args:
        df:
            A dataframe.
            Must have one of the combinations of keys:
            - 'start', 'end', 'enter', 'exit', 'low','high'
            - 'start', 'enter', 'exit', 'low', 'high'


    Returns:
        A `Candle` instance, type dependent on the input keys.
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


def _try_init_with_type_and_keys(df: pd.DataFrame, typ: type[Candle], *keys: str):
    if not all(k in df.columns for k in keys):
        return None

    dicts = {key: torch.tensor(df[key].tolist()) for key in keys}
    return typ(**dicts)
