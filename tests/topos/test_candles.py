# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."


from brokrest.topos import Candle


def _tensor_chart():
    "A randomly generated `CandleChart`."

    enter = torch.rand(100)
    exit = torch.rand(100)
    start = torch.randn(100)
    end = start + 1
    low = torch.zeros(100)
    high = torch.ones(100)

    yield BothCandle(enter=enter, exit=exit, start=start, end=end, low=low, high=high)
    yield LeftCandle(enter=enter, exit=exit, start=start, low=low, high=high)


def test_chart_is_sorted(candle: Candle):
    start = candle.left.tolist()
    assert list(start) == sorted(start)


def test_chart_index(candle: Candle):
    "Test chart's index access"

    assert len(candle) == 100
    assert isinstance(candle[10], Candle)
    assert isinstance(candle[99], Candle)
    assert isinstance(candle[0], Candle)
    assert candle[0].ndim == 0

    assert isinstance(candle[:10], Candle)
    assert candle[:10].ndim == 1
    assert len(candle[:10]) == 10
