# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for linear regressions."

import pytest
from pytest import FixtureRequest

from brokrest.rulers import BoundaryRulerLineReg, LineReg
from brokrest.topos import Candle, Line, Point


def _strategy():
    yield "enter-exit"
    yield "low-high"


@pytest.fixture(params=_strategy())
def points(candle: Candle, request: FixtureRequest) -> Point:
    return candle.points(request.param)


@pytest.fixture(params=[False, True])
def linreg(request: FixtureRequest) -> LineReg:
    return LineReg(bias=request.param)


def test_linreg(linreg: LineReg, points: Point):
    out = linreg(points)
    assert isinstance(out, Line)


def _boundary_ruler_linreg():
    yield BoundaryRulerLineReg()
    yield BoundaryRulerLineReg(rotate=True)
    yield BoundaryRulerLineReg(rotate=True, shift_ratio=0.3)


@pytest.fixture(params=_boundary_ruler_linreg())
def boundary_linreg(request: FixtureRequest) -> BoundaryRulerLineReg:
    return request.param


def test_boundary_linreg(boundary_linreg: LineReg, points: Point):
    out = boundary_linreg(points)
    assert isinstance(out, Line)
