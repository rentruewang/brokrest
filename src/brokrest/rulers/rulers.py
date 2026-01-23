# Copyright (c) The BrokRest Authors - All Rights Reserved

import abc
from typing import Protocol

from brokrest.topos import Point, Topo

__all__ = ["Ruler", "PencilCase"]


class Ruler(Protocol):
    "A ruler is a protocol that generates from a set of points, a ``Topo``."

    @abc.abstractmethod
    def __call__(self, chart: Point, /) -> Topo:
        """
        Generate

        Args:
            chart:
                The input chart. Currently they are ``Point``s for convenience.
                Would be switched to ``Candle`` once ready.

        Returns:
            The output topology.
        """

        ...


RULERS: dict[str, Ruler] = {}


def register_ruler(name: str):
    "Register a ruler into a global dictionary, s.t. it can be called by name."

    def register[R: Ruler](ruler: R) -> R:
        if name in RULERS:
            raise KeyError(f"Ruler with {name=} already exists.")

        RULERS[name] = ruler
        return ruler

    return register


class PencilCase(Protocol):
    "A pencil case may contain multiple rulers."

    @abc.abstractmethod
    def __call__(self, chart: Point, /) -> list[Topo]:
        """
        Generate a series of points from a single chart.

        Args:
            chart:
                The input chart. Currently they are ``Point``s for convenience.
                Would be switched to ``Candle`` once ready.


        Returns:
            The output topologies.
        """
