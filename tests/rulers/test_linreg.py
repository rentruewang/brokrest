# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for linear regressions."

import pytest
import torch
from pytest import FixtureRequest

from brokrest.rulers import LineReg
from brokrest.rulers.linear import LineReg
from brokrest.topos import Point


@pytest.fixture
def points() -> Point:
    return Point.init(x=torch.randn(100), y=torch.randn(100))


@pytest.fixture(params=[False, True])
def linreg(request: FixtureRequest) -> LineReg:
    return LineReg(bias=request.param)
