# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls

import numpy as np
import pytest
from numpy import random

from brokrest.arrays import ArrayDict


@dcls.dataclass
class Point(ArrayDict):
    x: np.ndarray
    y: np.ndarray


@dcls.dataclass
class PointMark(ArrayDict):
    point: Point
    mark: np.ndarray


@pytest.fixture
def point():
    return Point(x=random.randn(4), y=random.randn(4))


@pytest.fixture
def point_mark(point: Point):
    return PointMark(point=point, mark=random.randn(4))


def test_point(point: Point):
    assert point.shape == (4,)
    assert point.ndim == 1
    assert point.size == 4


def test_point_reshape(point: Point):
    reshaped = point.reshape(2, 2)
    assert reshaped.shape == (2, 2)


def test_point_expand(point: Point):
    p = point[None, ..., None]
    assert p.shape == (1, 4, 1)


def test_point_item(point: Point):
    p = point[0].item()
    assert isinstance(p.x, np.generic)
    assert isinstance(p.y, np.generic)


def test_point_dtype(point: Point):
    dt = point.dtype
    assert isinstance(dt, np.dtype)
