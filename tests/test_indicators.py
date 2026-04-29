# Copyright (c) The BrokRest Authors - All Rights Reserved

import numpy as np
import pytest
from numpy import random as npr

from brokrest.indicators import BollingerBand, Ema, Indicator, Macd, Rsi


@pytest.fixture
def data():
    return npr.randn(1000)


@pytest.fixture
def rsi():
    return Rsi()


@pytest.fixture
def ema():
    return Ema()


@pytest.fixture
def bollinger():
    return BollingerBand()


@pytest.fixture
def macd():
    return Macd()


@pytest.fixture(
    params=[
        rsi.__name__,
        ema.__name__,
        bollinger.__name__,
        macd.__name__,
    ],
)
def signal(request: pytest.FixtureRequest) -> Indicator:
    return request.getfixturevalue(request.param)


def test_signal_working(data: np.ndarray, signal: Indicator):
    out = signal(data)
    assert isinstance(out, np.ndarray)
    assert out.shape[-1] == len(data)
    assert len(out) == len(out)
