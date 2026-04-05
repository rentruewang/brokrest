# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for linear regressions."

import pytest
import torch

from brokrest.rulers import LineReg
from brokrest.topos import Point


@pytest.fixture
def points() -> Point:
    return Point(x=torch.randn(100), y=torch.randn(100))


@pytest.fixture(params=[False, True])
def linreg(request: pytest.FixtureRequest) -> LineReg:
    return LineReg(bias=request.param)
