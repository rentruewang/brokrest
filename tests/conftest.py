# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."

import numpy as np
import pandas as pd
import pytest
from numpy import random

from brokrest.topos import BothCandle, Candle, LeftCandle, candles


def _candle_chart():
    "A randomly generated ``CandleChart``."

    enter = random.rand(100)
    exit = random.rand(100)
    start = random.randn(100)
    end = start + 1
    low = random.zeros(100)
    high = random.ones(100)

    yield BothCandle(enter=enter, exit=exit, start=start, end=end, low=low, high=high)
    yield LeftCandle(enter=enter, exit=exit, start=start, low=low, high=high)


def _dataframe_chart():
    enter = random.rand(100)
    exit = random.rand(100)
    start = random.randn(100)
    end = start + 1
    low = np.zeros(100)
    high = np.ones(100)

    yield candles.dataframe_to_candles(
        pd.DataFrame(
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
    yield candles.dataframe_to_candles(
        pd.DataFrame(
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
    yield from _candle_chart()
    yield from _dataframe_chart()


@pytest.fixture(params=_candles())
def candle(request: pytest.FixtureRequest) -> Candle:
    return request.param
