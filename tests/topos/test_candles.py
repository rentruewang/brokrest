# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."

import numpy as np
import pandas as pd
import pytest
from numpy import random

from brokrest.topos import BothCandle, Candle, LeftCandle
from brokrest.topos.candles import dataframe_to_candles


def _candle_chart():
    "A randomly generated `CandleChart`."

    enter = random.rand(100)
    exit = random.rand(100)
    start = random.randn(100)
    end = start + 1
    low = np.zeros(100)
    high = np.ones(100)

    yield BothCandle(enter=enter, exit=exit, start=start, end=end, low=low, high=high)
    yield LeftCandle(enter=enter, exit=exit, start=start, low=low, high=high)


def _dataframe_chart():
    enter = random.rand(100)
    exit = random.rand(100)
    start = random.randn(100)
    end = start + 1
    low = np.zeros(100)
    high = np.ones(100)

    yield dataframe_to_candles(
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
    yield dataframe_to_candles(
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


def _chart():
    yield from _candle_chart()
    yield from _dataframe_chart()


@pytest.fixture(params=_chart())
def chart(request: pytest.FixtureRequest) -> Candle:
    return request.param


def test_chart_is_1d(chart: Candle):
    assert chart.ndim == 1


def test_chart_index(chart: Candle):
    "Test chart's index access"

    assert len(chart) == 100
    assert isinstance(chart[10], Candle)
    assert isinstance(chart[99], Candle)
    assert isinstance(chart[0], Candle)
    assert chart[0].ndim == 0

    assert isinstance(chart[:10], Candle)
    assert chart[:10].ndim == 1
    assert len(chart[:10]) == 10


def test_boundary(chart: Candle):
    convex = chart.convex()
    assert convex is not None
    assert convex.ndim == 0


def test_where(chart: Candle):
    selected = chart.where(0, 1)
    assert isinstance(selected, Candle)
