# Copyright (c) The BrokRest Authors - All Rights Reserved

"The canvas that handles plotting."

import abc
import dataclasses as dcls
import functools
import typing

from bokeh import plotting

__all__ = ["ViewPort", "Displayable"]


@dcls.dataclass(frozen=True)
class ViewPort:
    """
    `ViewPort` specifies the region where `Canvas` is plotting
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
        Whether or not the `ViewPort` requires user handling.
        The default (infinite size) does not require user handling, so would be `False`.
        All other configuration would be `True`.
        """

        return (
            False
            or self.left != -float("inf")
            or self.right != float("inf")
            or self.bottom != -float("inf")
            or self.top != float("inf")
        )

    @functools.cached_property
    def figure(self):
        return plotting.figure()

    def display(self, display: "Displayable"):
        display.draw_on(self)
        return self

    def show(self):
        return plotting.show(self.figure)


@typing.runtime_checkable
class Displayable(typing.Protocol):
    """
    `Painter` paints on the `Canvas`.
    """

    @abc.abstractmethod
    def draw_on(self, vp: ViewPort, /) -> None:
        """
        Each painter should decide how to paint on `Canvas`,
        with the supported methods.

        Args:
            canvas: Canvas to invoke. Should be invoked sequentially.
        """

        raise NotImplementedError
