# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for linear regressions."

import pytest
import torch

from brokrest import rulers, topos


@pytest.fixture
def points() -> topos.Point:
    return topos.Point(x=torch.randn(100), y=torch.randn(100))


@pytest.fixture(params=[False, True])
def linreg(request: pytest.FixtureRequest) -> rulers.LineReg:
    return rulers.LineReg(bias=request.param)
