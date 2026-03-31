# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for equations."

import typing

import pytest
import torch

from brokrest import topos


class _LinearEqSolve(typing.NamedTuple):
    eq: topos.Line
    point: topos.Point


def _solve_cases():
    # Test intercept forms
    yield _LinearEqSolve(
        eq=topos.Line.intercept(
            a=torch.tensor(5),
            b=torch.tensor(4),
        ),
        point=topos.Point(
            x=torch.tensor([5, 0]),
            y=torch.tensor([0, 4]),
        ),
    )

    # Test slope-intercept
    yield _LinearEqSolve(
        eq=topos.Line.slope_intercept(
            m=torch.tensor(9),
            b=torch.tensor(3),
        ),
        point=topos.Point(
            x=torch.tensor([0, 1]),
            y=torch.tensor([3, 12]),
        ),
    )


@pytest.mark.parametrize("case", _solve_cases())
def test_sub_cases_solved(case: _LinearEqSolve):
    "Points on the line (already solved cases) yields 0."

    assert torch.allclose(
        case.eq.subs(case.point).float(),
        torch.zeros([case.eq.numel(), case.point.numel()]).float(),
    )
