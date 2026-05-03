# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for equations."

import typing

import numpy as np
import pytest
from numpy import random

from brokrest.topos import Importance, Line, Point, Window


class _LinearEqSolve(typing.NamedTuple):
    eq: Line
    point: Point


def _solve_cases():
    # Test intercept forms
    yield _LinearEqSolve(
        eq=Line.intercept(a=np.array(5), b=np.array(4)),
        point=Point(x=np.array([5, 0]), y=np.array([0, 4])),
    )

    # Test slope-intercept
    yield _LinearEqSolve(
        eq=Line.slope_intercept(m=np.array(9), b=np.array(3)),
        point=Point(x=np.array([0, 1]), y=np.array([3, 12])),
    )


@pytest.mark.parametrize("case", _solve_cases())
def test_sub_cases_solved(case: _LinearEqSolve):
    "Points on the line (already solved cases) yields 0."

    assert np.allclose(case.eq.subs(case.point), b=0)


@pytest.fixture
def line():
    return Line.slope_intercept(np.array(1), np.array(2))


@pytest.fixture
def lines():
    return Line.slope_intercept(random.randn(5), random.randn(5))


@pytest.fixture
def point():
    return Point(1, 2)


@pytest.fixture
def points():
    return Point(random.randn(5), random.randn(5))


class DistTestCase(typing.NamedTuple):
    lines: Line
    points: Point
    shape: tuple[int, ...]


def _distance_cases():
    line = Line.slope_intercept(np.array(1), np.array(2))
    point = Point(np.array(3), np.array(4))
    lines = Line.slope_intercept(random.randn(5), random.randn(5))
    points = Point(random.randn(6), random.randn(6))

    yield DistTestCase(line, point, ())
    yield DistTestCase(lines, point, (5,))
    yield DistTestCase(line, points, (6,))
    yield DistTestCase(lines, points, (5, 6))


@pytest.fixture(params=_distance_cases())
def dist_case(request: pytest.FixtureRequest):
    return request.param


def _dist_funcs():
    yield Window(-1, 1)
    yield Window(0.5, 0.5)
    yield Window(-float("inf"), float("inf"))


@pytest.fixture(params=_dist_funcs())
def dist_func(request: pytest.FixtureRequest):
    return request.param


def test_distance(dist_case: DistTestCase, dist_func: Importance):
    line, point, shape = dist_case

    dist = line.dist(point)
    assert dist.shape == shape

    score = line.dist_loss_score(point, dist_func)
    assert score >= 0
