# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np
import talib
import torch
from numpy import typing as npt

__all__ = ["Indicator", "Rsi", "Ema", "Macd", "BollingerBand"]


FloatArray = npt.NDArray[np.float64]


class Indicator(typing.Protocol):
    """
    `Indicator` is a callable that converts the raw datapoint into some signals.
    It must have the same number of datapoints, matching the original input.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.ta_lib(data.cpu().numpy().astype("float64")))

    def ta_lib(self, data: FloatArray, /) -> FloatArray: ...


@dcls.dataclass(frozen=True)
class Rsi(Indicator):
    window: int = 14

    def __post_init__(self):
        if self.window <= 0:
            raise ValueError(f"{self.window=} should be a positive number.")

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        return talib.RSI(data, timeperiod=self.window)


@dcls.dataclass
class Ema(Indicator):
    period: float = 30

    def __post_init__(self):
        if not self.period > 0:
            raise ValueError(f"{self.period=}, but should be positive.")

    @typing.override
    def ta_lib(self, data: FloatArray, /) -> FloatArray:
        return talib.EMA(data, timeperiod=self.period)


@dcls.dataclass
class Macd(Indicator):
    """
    MACD signal is just EMA_fast - EMA_slow.
    """

    fast: float = 12
    slow: float = 26
    signal: float = 12

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
