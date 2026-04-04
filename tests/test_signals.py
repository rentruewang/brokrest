# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest
import torch

from brokrest import signals


@pytest.fixture
def data() -> torch.Tensor:
    return torch.randn(1000)


@pytest.fixture
def rsi():
    return signals.Rsi()


@pytest.fixture
def ema():
    return signals.Ema()


@pytest.fixture
def bollinger():
    return signals.BollingerBand()


@pytest.fixture
def macd():
    return signals.Macd()


@pytest.fixture(
    params=[
        rsi.__name__,
        ema.__name__,
        bollinger.__name__,
        macd.__name__,
    ],
)
def signal(request: pytest.FixtureRequest) -> signals.Signal:
    return request.getfixturevalue(request.param)


def test_signal_working(data: torch.Tensor, signal: signals.Signal):
    out = signal(data)
    assert isinstance(out, torch.Tensor)
    assert len(out) == len(out)
