# Copyright (c) The BrokRest Authors - All Rights Reserved

"The canvas that handles plotting."

import abc
import dataclasses as dcls
import typing
from typing import Protocol

from bokeh import plotting
from bokeh.plotting import figure as Figure

__all__ = ["Canvas", "Window", "Displayable"]


@dcls.dataclass(frozen=True)
class Window:
    """
    ``Window`` specifies the region where ``Canvas`` is plotting
    """

    left: float = -float("inf")
    right: float = float("inf")
    bottom: float = -float("inf")
    top: float = float("inf")

    def __post_init__(self) -> None:
        if self.left >= self.right or self.bottom >= self.top:
            raise ValueError(
                " ".join(
                    [
                        f"{self.left=} >= {self.right=} or {self.bottom=} >= {self.top=}",
                        "neither of them is allowed.",
                    ]
                )
            )

    def is_set(self) -> bool:
        """
        Whether or not the ``ViewPort`` requires user handling.
        The default (infinite size) does not require user handling, so would be ``False``.
        All other configuration would be ``True``.
        """

        return (
            False
            or self.left != -float("inf")
            or self.right != float("inf")
            or self.bottom != -float("inf")
            or self.top != float("inf")
        )


@dcls.dataclass(frozen=True)
class Canvas:
    """
    The canvas to plot on.
    """

    window: Window
    """
    The viewport to use.
    """

    figure: Figure
    "The figure to plot on."

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
