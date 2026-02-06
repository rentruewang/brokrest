# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for candles."


from brokrest.topos import Candle


def test_chart_is_1d(candle: Candle):
    assert candle.ndim == 1


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
