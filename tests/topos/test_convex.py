# Copyright (c) The BrokRest Authors - All Rights Reserved

import numpy as np
import pytest

from brokrest.data.yquery import load_yahooquery
from brokrest.topos import Candle, Point, Polygon, Segment


@pytest.fixture(params=[False, True])
def enter_exit(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def candle():
    return load_yahooquery(interval="1h")


@pytest.fixture
def convex_hull(candle: Candle, enter_exit: bool):
    pytest.xfail("Jagged array support")
    return candle.convex(enter_exit)


@pytest.fixture
def all_coords(candle: Candle, enter_exit: bool):
    return candle.top_bottom_bounds(enter_exit).flatten()


@pytest.fixture
def outer_shell_points(convex_hull: Polygon, all_coords: Point):
    outer_filter = all_coords.is_vertex_of(convex_hull)
    return all_coords[outer_filter]


def test_convex_points(outer_shell_points: Point, convex_hull: Polygon):
    # Ensure all points should have at least a match.
    assert outer_shell_points.is_vertex_of(convex_hull).all().item()


@pytest.fixture
def polygon():
    pytest.xfail("Jagged array support")
    return Polygon.from_vertices(
        Point(np.array(0), np.array(0)),
        Point(np.array(0), np.array(1)),
        Point(np.array(0.5), np.array(1.02)),
        Point(np.array(1), np.array(1)),
        Point(np.array(1), np.array(0)),
    )


def test_segments_merge(polygon: Polygon):
    assert len(polygon.segments) == 5
    reduced = polygon.segments.merge_similar_mono()
    assert isinstance(reduced, Segment)
    assert len(reduced) == 4
