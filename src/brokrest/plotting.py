# Copyright (c) The BrokRest Authors - All Rights Reserved

"The canvas that handles plotting."

import abc
import dataclasses as dcls
import typing
from typing import Protocol

from bokeh import plotting
from bokeh.plotting import figure as Figure

__all__ = ["Canvas", "Displayable"]


@dcls.dataclass(frozen=True)
class Canvas:
    """
    The canvas to plot on.
    """

    starting: float
    "The left most boundary."

    ending: float
    "The right most boundary."

    figure: Figure
    "The figure to plot on."

    def __post_init__(self) -> None:
        if self.starting >= self.ending:
            raise ValueError(
                f"{self.starting=} >= {self.ending=} which is not allowed."
            )

    def show(self):
        plotting.show(self.figure)


@typing.runtime_checkable
class Displayable(Protocol):
    """
    ``Painter`` paints on the ``Canvas``.
    """

    @abc.abstractmethod
    def draw(self, canvas: Canvas, /) -> None:
        """
        Each painter should decide how to paint on ``Canvas``,
        with the supported methods.

        Args:
            canvas: Canvas to invoke. Should be invoked sequentially.
        """

        ...
