# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."

import numpy as np
import pytest
import torch
from numpy import random
from pandas import DataFrame

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


def _chart():
    yield from _tensor_chart()
    yield from _dataframe_chart()


@pytest.fixture(params=_chart())
def chart(request: pytest.FixtureRequest) -> Candle:
    return request.param


def test_chart_is_1d(chart: Candle):
    assert chart.ndim == 1


def test_chart_is_sorted(chart: Candle):
    start = chart.left.tolist()
    assert list(start) == sorted(start)


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
