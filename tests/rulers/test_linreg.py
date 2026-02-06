# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for linear regressions."

import pytest
from pytest import FixtureRequest

from brokrest.rulers import LineReg
from brokrest.topos import Candle, Point


def _strategy():
    yield "enter-exit"
    yield "low-high"


@pytest.fixture(params=_strategy())
def points(candle: Candle, request: FixtureRequest) -> Point:
    return candle.points(request.param)


@pytest.fixture(params=[False, True])
def linreg(request: FixtureRequest) -> LineReg:
    return LineReg(bias=request.param)
