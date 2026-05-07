# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls

import numpy as np
import pytest
from numpy import random

from brokrest.arrays import ArrayDict


@dcls.dataclass(frozen=True)
class Point(ArrayDict):
    x: np.ndarray
    y: np.ndarray


@dcls.dataclass(frozen=True)
class PointMark(ArrayDict):
    point: Point
    mark: np.ndarray


@pytest.fixture
def point():
    return Point(x=random.randn(3), y=random.randn(3))


def test_point(point: Point):
    assert point.shape == (3,)
    assert point.ndim == 1
    assert point.size == 3


def test_point_reshape(point: Point):
    pass


print(point)
print(point[None, 1:].shape)

print(point.dtype)
point_mark = PointMark(point, random.randn(3))
print(point_mark)
print(point_mark.dtype)
