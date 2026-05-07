# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest
from numpy import random

from brokrest.topos import Point, Polygon, Segment


def _segments():
    return Segment(
        x_0=random.randn(13),
        x_1=random.randn(13),
        y_0=random.randn(13),
        y_1=random.randn(13),
    )


@pytest.fixture
def segments():
    return _segments()


def _points(segments: Segment):
    return segments.points()


@pytest.fixture
def points(segments: Segment):
    return _points(segments)


def _polygons():
    segments = _segments()
    points = _points(segments)
    yield Polygon.from_vertices(points)


def test_segments_shape(segments: Segment):
    assert segments.ndim == 1


def test_points_ndim(points: Point):
    assert points.ndim == 1


@pytest.fixture(params=_polygons())
def polygon(request: pytest.FixtureRequest):
    return request.param


def test_polygon_shape(polygon: Polygon):
    assert polygon.ndim == 0
