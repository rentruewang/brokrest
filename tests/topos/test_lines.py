# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for equations."

import typing

import pytest
import torch

from brokrest.topos import Line, Point


class _LinearEqSolve(typing.NamedTuple):
    eq: Line
    point: Point


def _solve_cases():
    # Test intercept forms
    yield _LinearEqSolve(
        eq=Line.intercept(
            a=torch.tensor(5),
            b=torch.tensor(4),
        ),
        point=Point(
            x=torch.tensor([5, 0]),
            y=torch.tensor([0, 4]),
        ),
    )

    # Test slope-intercept
    yield _LinearEqSolve(
        eq=Line.slope_intercept(
            m=torch.tensor(9),
            b=torch.tensor(3),
        ),
        point=Point(
            x=torch.tensor([0, 1]),
            y=torch.tensor([3, 12]),
        ),
    )


@pytest.mark.parametrize("case", _solve_cases())
def test_sub_cases_solved(case: _LinearEqSolve):
    "Points on the line (already solved cases) yields 0."

    assert torch.allclose(
        case.eq.subs(case.point).float(),
        torch.zeros([case.eq.numel(), case.point.numel()]).float(),
    )


@pytest.fixture
def line():
    return Line(1, 2)


@pytest.fixture
def lines():
    return Line(torch.randn(5), torch.randn(5))


@pytest.fixture
def point():
    return Point(1, 2)


@pytest.fixture
def points():
    return Point(torch.randn(5), torch.randn(5))


class DistTestCase(typing.NamedTuple):
    lines: Line
    points: Point
    shape: tuple[int, ...]


def _distance_cases():
    line = Line(1, 2)
    point = Point(3, 4)
    lines = Line(torch.randn(5), torch.randn(5))
    points = Point(torch.randn(6), torch.randn(6))

    yield DistTestCase(line, point, ())
    yield DistTestCase(lines, point, (5,))
    yield DistTestCase(line, points, (6,))
    yield DistTestCase(lines, points, (5, 6))


@pytest.fixture(params=_distance_cases())
def dist_case(request: pytest.FixtureRequest):
    return request.param


def test_distance(dist_case: DistTestCase):
    line, point, shape = dist_case

    dist = line.distance(point)
    assert dist.shape == shape
    assert (dist >= 0).all()
