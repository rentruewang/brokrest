# Copyright (c) The BrokRest Authors - All Rights Reserved

import dataclasses as dcls

import torch

from brokrest.topos import (
    Line,
    Point,
    Segment,
    Topo,
    TopoHandlerBase,
    enabled_topo_handlers,
)


@dcls.dataclass(frozen=True)
class Logger(TopoHandlerBase):
    topos: list[Topo] = dcls.field(default_factory=list)

    def __repr__(self) -> str:
        return repr([type(s) for s in self.topos])

    def __call__(self, topo):
        self.topos.append(topo)


def test_topo_handler():
    with Logger([]).enable() as l:
        Segment(1, 2, 3, 4)
        Point(1, 2)
        Line(torch.randn(5), torch.randn(5))

        assert len(enabled_topo_handlers()) == 1
        assert enabled_topo_handlers()[0] is l

    assert not enabled_topo_handlers()
