# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor

from brokrest.signals import BollingerBand, Ema, Macd, Rsi, Signal


@pytest.fixture
def data() -> Tensor:
    return torch.randn(1000)


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
def signal(request: FixtureRequest) -> Signal:
    return request.getfixturevalue(request.param)


def test_signal_working(data: Tensor, signal: Signal):
    out = signal(data)
    assert isinstance(out, Tensor)
    assert len(out) == len(out)
