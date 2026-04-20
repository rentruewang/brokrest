# Copyright (c) The BrokRest Authors - All Rights Reserved


import torch

from brokrest.topos import (
    Line,
    Point,
    Segment,
    TopoInScope,
    enabled_topo_handlers,
)


def test_topo_handler():
    with TopoInScope([]).enable() as topos:
        Segment(1, 2, 3, 4)
        Point(1, 2)
        Line.slope_intercept(torch.randn(5), torch.randn(5))

        assert len(enabled_topo_handlers()) == 1
        assert enabled_topo_handlers()[0] is topos
        assert len(topos) == 3

    assert not enabled_topo_handlers()
