# Copyright (c) The BrokRest Authors - All Rights Reserved

import numpy as np
from numpy import random

from brokrest.topos import (
    Line,
    Point,
    Segment,
    TopoInScope,
    enabled_topo_handlers,
)


def test_topo_handler():
    with TopoInScope([]).enable() as topos:
        Segment(np.array(1), np.array(2), np.array(3), np.array(4))
        Point(np.array(1), np.array(2))
        Line.slope_intercept(random.randn(5), random.randn(5))

        assert len(enabled_topo_handlers()) == 1
        assert enabled_topo_handlers()[0] is topos
        assert len(topos) == 3

    assert not enabled_topo_handlers()
