# Copyright (c) The BrokRest Authors - All Rights Reserved

"Tests for the 2D vector."

import operator
from collections.abc import Callable
from typing import NamedTuple

import pytest
from pytest import FixtureRequest

from brokrest import vectors
from brokrest.vectors import Vec2d


def test_flip():
    "Test the flipping method."

    rsub = vectors.flip(operator.sub)

    assert operator.sub(3, 1) == 2
    assert rsub(3, 1) == -2


def test_eq():
    "The ``Vec2d``'s equal method."

    left = Vec2d(11, 13)
    right = Vec2d(11, 13)

    assert left is not right
    assert left == right


class _BinaryInput(NamedTuple):
    left: Vec2d | float
    right: Vec2d | float
    op: Callable[..., Vec2d]
    result: Vec2d


def _get_binary_pairs():
    # Test the + operator.

    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=Vec2d(6, 4),
        op=operator.add,
        result=Vec2d(9, 9),
    )
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=1,
        op=operator.add,
        result=Vec2d(4, 6),
    )
    yield _BinaryInput(
        left=1,
        right=Vec2d(3, 5),
        op=operator.add,
        result=Vec2d(4, 6),
    )

    # Test the - operator.
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=Vec2d(6, 4),
        op=operator.sub,
        result=Vec2d(-3, 1),
    )
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=1,
        op=operator.sub,
        result=Vec2d(2, 4),
    )
    yield _BinaryInput(
        left=1,
        right=Vec2d(3, 5),
        op=operator.sub,
        result=Vec2d(-2, -4),
    )

    # Test the * operator.
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=Vec2d(6, 4),
        op=operator.mul,
        result=Vec2d(18, 20),
    )
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=7,
        op=operator.mul,
        result=Vec2d(21, 35),
    )
    yield _BinaryInput(
        left=7,
        right=Vec2d(3, 5),
        op=operator.mul,
        result=Vec2d(21, 35),
    )

    # Test the / operator.
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=Vec2d(6, 4),
        op=operator.truediv,
        result=Vec2d(1 / 2, 5 / 4),
    )
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=2,
        op=operator.truediv,
        result=Vec2d(3 / 2, 5 / 2),
    )
    yield _BinaryInput(
        left=2,
        right=Vec2d(3, 5),
        op=operator.truediv,
        result=Vec2d(2 / 3, 2 / 5),
    )

    # Test the // operator.
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=Vec2d(6, 4),
        op=operator.floordiv,
        result=Vec2d(1 // 2, 5 // 4),
    )
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=2,
        op=operator.floordiv,
        result=Vec2d(3 // 2, 5 // 2),
    )
    yield _BinaryInput(
        left=4,
        right=Vec2d(3, 5),
        op=operator.floordiv,
        result=Vec2d(4 // 3, 4 // 5),
    )

    # Test the ** operator.
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=Vec2d(6, 4),
        op=operator.pow,
        result=Vec2d(3**6, 5**4),
    )
    yield _BinaryInput(
        left=Vec2d(3, 5),
        right=2,
        op=operator.pow,
        result=Vec2d(3**2, 5**2),
    )
    yield _BinaryInput(
        left=2,
        right=Vec2d(3, 5),
        op=operator.pow,
        result=Vec2d(2**3, 2**5),
    )


@pytest.fixture(params=_get_binary_pairs())
def binary_inputs(request: FixtureRequest):
    "The binary input test cases."

    return request.param


def test_vector_binary_op(binary_inputs: _BinaryInput):
    "Test the vector's binary operator."

    left, right, op, result = binary_inputs
    assert op(left, right) == result


def test_origin():
    "Test the convenience method ``origin``."

    assert Vec2d.origin() == Vec2d(0, 0)


def _get_vec_slopes():
    class _VecSlope(NamedTuple):
        vec: Vec2d
        slope: float

    yield _VecSlope(Vec2d(3, 5), 5 / 3)

    yield _VecSlope(Vec2d(9, -5), -5 / 9)


@pytest.mark.parametrize("vec,slope", _get_vec_slopes())
def test_vec_slope(vec: Vec2d, slope: float):
    assert vec.slope() == slope
