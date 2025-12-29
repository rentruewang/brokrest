# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for equations."

import math
from typing import NamedTuple

import pytest

from brokrest.equations import InterceptForm, LinearEq, SlopeInterceptForm, StandardForm


class _LinearEqSolve(NamedTuple):
    eq: LinearEq
    x: float
    y: float


def _solve_cases():
    # Standard forms.
    yield _LinearEqSolve(
        eq=StandardForm(a=1, b=1, c=1),
        x=1,
        y=-2,
    )
    yield _LinearEqSolve(
        eq=StandardForm(a=2, b=-1, c=4),
        x=2,
        y=8,
    )
    yield _LinearEqSolve(
        eq=StandardForm(a=2, b=-9, c=4),
        x=3,
        y=10 / 9,
    )

    # Test intercept forms
    yield _LinearEqSolve(
        eq=InterceptForm(a=5, b=4),
        x=5,
        y=0,
    )
    yield _LinearEqSolve(
        eq=InterceptForm(a=5, b=4),
        x=0,
        y=4,
    )

    # Test slope-intercept
    yield _LinearEqSolve(
        eq=SlopeInterceptForm(m=9, b=3),
        x=0,
        y=3,
    )
    yield _LinearEqSolve(
        eq=SlopeInterceptForm(m=9, b=3),
        x=1,
        y=12,
    )


@pytest.mark.parametrize("case", _solve_cases())
def test_linear_eq_solve(case: _LinearEqSolve):
    "Test the solving of ``LinearEq``."

    assert math.isclose(case.eq.solve(case.x), case.y)


@pytest.mark.parametrize("case", _solve_cases())
def test_sub_cases_solved(case: _LinearEqSolve):
    "Points on the line (already solved cases) yields 0."

    assert math.isclose(case.eq.subs(case.x, case.y), 0)
