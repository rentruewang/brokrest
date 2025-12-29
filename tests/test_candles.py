# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."

import pytest
from numpy import random

from brokrest.candles import Candle, CandleChart


@pytest.fixture()
def chart():
    "A randomly generated ``CandleChart``."

    return CandleChart.from_values((random.random([101]) * 100).tolist())


def test_chart_index(chart: CandleChart):
    "Test chart's index access"

    assert len(chart) == 100
    assert isinstance(chart[10], Candle)
    assert isinstance(chart[99], Candle)
    assert isinstance(chart[0], Candle)


def test_chart_contiguous_candles(chart: CandleChart):
    "Test if each candle has its end equal to the next one's start."

    time = chart.start

    # Check if all the start and end are connected.
    for candle in chart:
        assert time == candle.start

        time = candle.end
