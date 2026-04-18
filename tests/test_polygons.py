# Copyright (c) The BrokRest Authors - All Rights Reserved

import pytest
from brokrest.topos import Point
from brokrest.topos import Polygon
import torch


@pytest.fixture
def points():
    return Point(torch.randn(13, 11), torch.randn(13, 11))


@pytest.fixture
def polygon(points: Point):
    return Polygon.from_vertices(points)


def test_polygon_shape(polygon: Polygon):
    assert polygon.shape == (13,)
    assert polygon.ndim == 1
