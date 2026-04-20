# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest

from brokrest.data.yquery import load_yahooquery
from brokrest.topos import Candle, Point, Polygon


@pytest.fixture(params=[False, True])
def enter_exit(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def candle():
    return load_yahooquery(interval="1h")


@pytest.fixture
def convex_hull(candle: Candle, enter_exit: bool):
    return candle.convex(enter_exit)


@pytest.fixture
def all_coords(candle: Candle, enter_exit: bool):
    return candle.coords(enter_exit)


@pytest.fixture
def outer_shell_points(convex_hull: Polygon, all_coords: Point):
    outer_filter = all_coords.is_vertex_of(convex_hull)
    return all_coords[outer_filter]


def test_convex_points(outer_shell_points: Point, convex_hull: Polygon):
    eq = outer_shell_points.cross_eq_1d(convex_hull.vertices)

    # Ensure all points should have at least a match.
    assert outer_shell_points.is_vertex_of(convex_hull).all().item()
