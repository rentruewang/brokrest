# Copyright (c) The BrokRest Authors - All Rights Reserved

import abc
import typing

from brokrest import topos

__all__ = ["Ruler", "PencilCase"]


class Ruler(typing.Protocol):
    "A ruler is a protocol that generates from a set of points, a `topos.Topo`."

    @abc.abstractmethod
    def __call__(self, chart: topos.Point, /) -> topos.Topo:
        """
        Generate

        Args:
            chart:
                The input chart. Currently they are `topos.Point`s for convenience.
                Would be switched to `Candle` once ready.

        Returns:
            The output topology. Since `topos.Topo` can be batched,
            the output can represent multiple homogenius topologies.
        """

        ...


class PencilCase(typing.Protocol):
    "A pencil case may contain multiple rulers."

    @abc.abstractmethod
    def __call__(self, chart: topos.Point, /) -> list[topos.Topo]:
        """
        Generate a series of points from a single chart.

        Args:
            chart:
                The input chart. Currently they are `topos.Point`s for convenience.
                Would be switched to `Candle` once ready.


        Returns:
            The output is a set of (maybe) heterogenius topologies.
        """
