# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."

import pytest
import torch

from brokrest.topos import Candle


@pytest.fixture()
def chart():
    "A randomly generated ``CandleChart``."

    enter = torch.rand(100)
    exit = torch.rand(100)
    start = torch.randn(100)
    end = start + 1
    low = torch.zeros(100)
    high = torch.ones(100)
    return Candle(enter=enter, exit=exit, start=start, end=end, low=low, high=high)


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
