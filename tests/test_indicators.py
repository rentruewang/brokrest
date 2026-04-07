# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest
import torch

from brokrest.indicators import BollingerBand, Ema, Indicator, Macd, Rsi, convolve


@pytest.fixture
def data() -> torch.Tensor:
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
def signal(request: pytest.FixtureRequest) -> Indicator:
    return request.getfixturevalue(request.param)


def test_signal_working(data: torch.Tensor, signal: Indicator):
    out = signal(data)
    assert isinstance(out, torch.Tensor)
    assert len(out) == len(out)


def test_convolve_1d():
    a = torch.randn(9)
    b = torch.randn(10)
    assert convolve(a, b).ndim == 1
