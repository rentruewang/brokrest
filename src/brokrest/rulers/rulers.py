# Copyright (c) The BrokRest Authors - All Rights Reserved

import abc
from typing import Protocol

from brokrest.topos import Point, Topo,Line

__all__ = ["Ruler", "PencilCase"]


class Ruler(Protocol):
    "A ruler is a protocol that generates from a set of points, a ``Topo``."

    @abc.abstractmethod
    def __call__(self, chart: Point, /) -> Line:
        """
        Generate

        Args:
            chart:
                The input chart. Currently they are ``Point``s for convenience.
                Would be switched to ``Candle`` once ready.

        Returns:
            The output topology. Since ``Topo`` can be batched,
            the output can represent multiple homogenius topologies.
        """

        ...


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
            The output is a set of (maybe) heterogenius topologies.
        """
