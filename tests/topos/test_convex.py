# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest

from brokrest.data.yquery import load_yahooquery
from brokrest.topos import Candle, Point, Polygon


@pytest.fixture
def candle():
    return load_yahooquery(interval="1h")


@pytest.fixture
def convex_hull(candle: Candle):
    return candle.convex()


@pytest.fixture
def all_coords(candle: Candle):
    return candle.coords()


@pytest.fixture
def outer_shell_points(convex_hull: Polygon, all_coords: Point):
    outer_filter = all_coords.cross_eq_1d(convex_hull.vertices).any(dim=1)
    return all_coords[outer_filter]


def test_convex_points_working(outer_shell_points: Point, convex_hull: Polygon):
    eq = outer_shell_points.cross_eq_1d(convex_hull.vertices)

    # Use .any to find out whether there is a match.
    # Use .all to ensure all points should have at least a match.
    assert eq.any(dim=1).all().item()
