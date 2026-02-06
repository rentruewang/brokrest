# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for linear regressions."

import pytest
from pytest import FixtureRequest

from brokrest.rulers import LineReg
from brokrest.topos import Candle, Point


@pytest.fixture
def points() -> Point:
    return Point(x=torch.randn(100), y=torch.randn(100))


@pytest.fixture(params=[False, True])
def linreg(request: FixtureRequest) -> LineReg:
    return LineReg(bias=request.param)
