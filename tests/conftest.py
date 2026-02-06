# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."

import numpy as np
import pytest
import torch
from numpy import random
from pandas import DataFrame
from pytest import FixtureRequest

from brokrest.topos import BothCandle, Candle, LeftCandle, candles


def _tensor_chart():
    "A randomly generated ``CandleChart``."

    enter = torch.rand(100)
    exit = torch.rand(100)
    start = torch.randn(100)
    end = start + 1
    low = torch.zeros(100)
    high = torch.ones(100)

    yield BothCandle.init(
        enter=enter, exit=exit, start=start, end=end, low=low, high=high
    )
    yield LeftCandle.init(enter=enter, exit=exit, start=start, low=low, high=high)


def _dataframe_chart():
    enter = random.rand(100)
    exit = random.rand(100)
    start = random.randn(100)
    end = start + 1
    low = np.zeros(100)
    high = np.ones(100)

    yield candles.dataframe_factory(
        DataFrame(
            {
                "enter": enter,
                "exit": exit,
                "start": start,
                "end": end,
                "low": low,
                "high": high,
            }
        )
    )
    yield candles.dataframe_factory(
        DataFrame(
            {
                "enter": enter,
                "exit": exit,
                "start": start,
                "low": low,
                "high": high,
            }
        )
    )


def _candles():
    yield from _tensor_chart()
    yield from _dataframe_chart()


@pytest.fixture(params=_candles())
def candle(request: FixtureRequest) -> Candle:
    return request.param
