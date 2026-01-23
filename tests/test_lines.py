# Copyright (c) The BrokRest Authors - All Rights Reserved

"Test cases for equations."

from typing import NamedTuple

import pytest
import torch

from brokrest.topos import Line, Point


class _LinearEqSolve(NamedTuple):
    eq: Line
    point: Point


def _solve_cases():
    # Test intercept forms
    yield _LinearEqSolve(
        eq=Line.intercept(
            a=torch.tensor(5),
            b=torch.tensor(4),
        ),
        point=Point.init_tensor(
            x=torch.tensor([5, 0]),
            y=torch.tensor([0, 4]),
        ),
    )

    # Test slope-intercept
    yield _LinearEqSolve(
        eq=Line.slope_intercept(
            m=torch.tensor(9),
            b=torch.tensor(3),
        ),
        point=Point.init_tensor(
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
