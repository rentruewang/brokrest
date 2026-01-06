# Copyright (c) The BrokRest Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Sequence
from typing import Self

from bokeh.plotting import figure as Figure

from brokrest.plotting import Canvas, Displayable, ViewPort

__all__ = ["Topo", "TopoSet"]


class Topo(Displayable, ABC):
    """
    A topology that can be displayed on the ``Canvas``.
    """

    @typing.override
    def draw(self, canvas: Canvas, /) -> None:
        sub = self.view(canvas.view)
        sub._draw(canvas.figure)

    @abc.abstractmethod
    def _draw(self, figure: Figure):
        """
        Paint on the figure.
        """

    def view(self, vp: ViewPort, /) -> "Topo":
        """
        Get the subset of topology within the range.
        """

        # Check if ``vp`` would require human intervension.
        if not vp:
            return self

        return self._cut(vp)

    @abc.abstractmethod
    def _cut(self, vp: ViewPort, /) -> "Topo":
        "The implementation of ``cut``."
        ...


@dcls.dataclass(frozen=True)
class TopoSet(Sequence[Topo], Topo):
    """
    ``TopoSet`` is a set of topologies. It is itself a ``Topo``.
    """

    topos: Sequence[Topo]
    """
    The sequence of ``Topo``s being held in the set.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.topos)

    @typing.overload
    def __getitem__(self, idx: int) -> Topo: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[int]) -> Self: ...

    @typing.override
    def __getitem__(self, idx):
        match idx:
            case int():
                return self.topos[idx]
            case slice():
                return type(self)(self.topos[idx])
            case list():
                return type(self)([self.topos[i] for i in idx])
            case _:
                raise NotImplementedError(f"Type: {type(idx)=} is not supported.")

    @typing.override
    def _cut(self, vp: ViewPort) -> "Topo":
        return type(self)([topo._cut(vp) for topo in self.topos])

    @typing.override
    def _draw(self, figure: Figure):
        for topo in self.topos:
            topo._draw(figure)
