# Copyright (c) The BrokRest Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np
import talib
from numpy import typing as npt

from brokrest.plotting import Displayable, ViewPort
from brokrest.topos import Candle

__all__ = ["Indicator", "IndicatorList", "Rsi", "Ema", "Macd", "BollingerBand"]


FloatArray: typing.TypeAlias = npt.NDArray[np.float64]


class Indicator(abc.ABC):
    """
    `Indicator` is a callable that converts the raw datapoint into some signals.
    It must have the same number of datapoints, matching the original input.

    In `compute`, the result can be either 1D (1 indicator) or 2D (N indicators).
    1D tensors would be casted to (1, length) tensors in `__call__`.
    """

    def __call__(self, data: FloatArray, /) -> FloatArray:
        if data.ndim != 1:
            raise ValueError(f"Input should be a 1D tensor. Got {data.shape=}.")

        result = self.ta_lib(data)

        match result.ndim:
            case 1:
                return np.expand_dims(result, 0)
            case 2:
                return result
            case _:
                raise AssertionError(
                    f"Expected 1D or 2D tensor output, got {result.shape=}."
                )

    @abc.abstractmethod
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class CandleIndicator(Displayable):
    candles: Candle
    indicator: Indicator

    @typing.override
    def draw_on(self, vp: ViewPort, /) -> None:
        selected = (self.candles.left >= vp.left) & (self.candles.right <= vp.right)
        filtered_candles: Candle = self.candles[selected]
        exit_values = self.candles.exit.cpu().numpy().astype("float64")
        times = self.candles.right.cpu().numpy().astype("float64")

        indicators = self.indicator(exit_values)

        vp.display(filtered_candles)

        for ind in indicators:
            vp.figure.segment(
                x0=times[:-1],
                x1=times[1:],
                y0=ind[:-1],
                y1=ind[1:],
            )


@dcls.dataclass(frozen=True)
class IndicatorList(Indicator):
    indicators: cabc.Sequence[Indicator]

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        results = np.concatenate([ind(data) for ind in self.indicators])
        return results


@dcls.dataclass(frozen=True)
class Rsi(Indicator):
    window: int = 14

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"{self.window=} should be a positive number.")

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        return talib.RSI(data, timeperiod=self.window)


@dcls.dataclass(frozen=True)
class Ema(Indicator):
    period: int = 30

    def __post_init__(self):
        if not self.period > 0:
            raise ValueError(f"{self.period=}, but should be positive.")

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        return talib.EMA(data, timeperiod=self.period)


@dcls.dataclass(frozen=True)
class Macd(Indicator):
    """
    MACD signal is just EMA_fast - EMA_slow.
    """

    fast: int = 12
    slow: int = 26
    signal: int = 12

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        result, _, _ = talib.MACD(
            data,
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal,
        )
        return result


@dcls.dataclass(frozen=True)
class BollingerBand(Indicator):
    """
    Bollinger band is a lower, middle, upper band.
    """

    window: int = 5
    num_std_up: float = 2
    num_std_down: float = 2

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        """
        Compute Bollinger Bands in PyTorch.
        """

        low, mid, top = talib.BBANDS(
            data,
            timeperiod=self.window,
            nbdevdn=self.num_std_down,
            nbdevup=self.num_std_up,
        )
        return np.stack([low, mid, top])
