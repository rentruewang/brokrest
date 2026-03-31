# Copyright (c) The BrokRest Authors - All Rights Reserved

"Tests for the 2D vector."

import operator
import typing
from collections import abc as cabc

import pytest

from brokrest import vectors


def test_flip():
    "Test the flipping method."

    rsub = vectors.flip(operator.sub)

    assert operator.sub(3, 1) == 2
    assert rsub(3, 1) == -2


def test_eq():
    "The `vectors.Vec2d`'s equal method."

    left = vectors.Vec2d(11, 13)
    right = vectors.Vec2d(11, 13)

    assert left is not right
    assert left == right


class _BinaryInput(typing.NamedTuple):
    left: vectors.Vec2d | float
    right: vectors.Vec2d | float
    op: cabc.Callable[..., vectors.Vec2d]
    result: vectors.Vec2d


def _get_binary_pairs():
    # Test the + operator.

    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=vectors.Vec2d(6, 4),
        op=operator.add,
        result=vectors.Vec2d(9, 9),
    )
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=1,
        op=operator.add,
        result=vectors.Vec2d(4, 6),
    )
    yield _BinaryInput(
        left=1,
        right=vectors.Vec2d(3, 5),
        op=operator.add,
        result=vectors.Vec2d(4, 6),
    )

    # Test the - operator.
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=vectors.Vec2d(6, 4),
        op=operator.sub,
        result=vectors.Vec2d(-3, 1),
    )
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=1,
        op=operator.sub,
        result=vectors.Vec2d(2, 4),
    )
    yield _BinaryInput(
        left=1,
        right=vectors.Vec2d(3, 5),
        op=operator.sub,
        result=vectors.Vec2d(-2, -4),
    )

    # Test the * operator.
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=vectors.Vec2d(6, 4),
        op=operator.mul,
        result=vectors.Vec2d(18, 20),
    )
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=7,
        op=operator.mul,
        result=vectors.Vec2d(21, 35),
    )
    yield _BinaryInput(
        left=7,
        right=vectors.Vec2d(3, 5),
        op=operator.mul,
        result=vectors.Vec2d(21, 35),
    )

    # Test the / operator.
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=vectors.Vec2d(6, 4),
        op=operator.truediv,
        result=vectors.Vec2d(1 / 2, 5 / 4),
    )
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=2,
        op=operator.truediv,
        result=vectors.Vec2d(3 / 2, 5 / 2),
    )
    yield _BinaryInput(
        left=2,
        right=vectors.Vec2d(3, 5),
        op=operator.truediv,
        result=vectors.Vec2d(2 / 3, 2 / 5),
    )

    # Test the // operator.
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=vectors.Vec2d(6, 4),
        op=operator.floordiv,
        result=vectors.Vec2d(1 // 2, 5 // 4),
    )
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=2,
        op=operator.floordiv,
        result=vectors.Vec2d(3 // 2, 5 // 2),
    )
    yield _BinaryInput(
        left=4,
        right=vectors.Vec2d(3, 5),
        op=operator.floordiv,
        result=vectors.Vec2d(4 // 3, 4 // 5),
    )

    # Test the ** operator.
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=vectors.Vec2d(6, 4),
        op=operator.pow,
        result=vectors.Vec2d(3**6, 5**4),
    )
    yield _BinaryInput(
        left=vectors.Vec2d(3, 5),
        right=2,
        op=operator.pow,
        result=vectors.Vec2d(3**2, 5**2),
    )
    yield _BinaryInput(
        left=2,
        right=vectors.Vec2d(3, 5),
        op=operator.pow,
        result=vectors.Vec2d(2**3, 2**5),
    )


@pytest.fixture(params=_get_binary_pairs())
def binary_inputs(request: pytest.FixtureRequest):
    "The binary input test cases."

    return request.param


def test_vector_binary_op(binary_inputs: _BinaryInput):
    "Test the vector's binary operator."

    left, right, op, result = binary_inputs
    assert op(left, right) == result


def test_origin():
    "Test the convenience method `origin`."

    assert vectors.Vec2d.origin() == vectors.Vec2d(0, 0)


def _get_vec_slopes():
    class _VecSlope(typing.NamedTuple):
        vec: vectors.Vec2d
        slope: float

    yield _VecSlope(vectors.Vec2d(3, 5), 5 / 3)

    yield _VecSlope(vectors.Vec2d(9, -5), -5 / 9)


@pytest.mark.parametrize("vec,slope", _get_vec_slopes())
def test_vec_slope(vec: vectors.Vec2d, slope: float):
    assert vec.slope() == slope
