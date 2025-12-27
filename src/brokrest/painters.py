# Copyright (c) The BrokRest Authors - All Rights Reserved

"The canvas that handles plotting."

import dataclasses as dcls

import seaborn as sns
from matplotlib import pyplot as plt

from .vectors import Vec2d


@dcls.dataclass(frozen=True)
class Box:
    """
    A box with 4 sides.
    """

    left: float
    "The left side."

    right: float
    "The right side."

    top: float
    "The top side."

    bottom: float
    "The bottom side."


@dcls.dataclass(frozen=True)
class Canvas:
    """
    The canvas to plot on.
    """

    left: int
    "The left most boundary."

    right: int
    "The right most boundary."

    def __post_init__(self):
        sns.set_theme()
        plt.clf()

    def fill(self, *, box: Box, color: str) -> None:
        """
        Draws a rectangular box on the canvas with the given coordinates, color, and optional fill.

        Args:
            box: The box to fill.
            color: Color of the box border.
        """

        plt.fill(
            [box.left, box.left, box.right, box.right],
            [box.top, box.bottom, box.bottom, box.top],
            color=color,
        )

    def border(
        self,
        *,
        box: Box,
        color: str,
    ) -> None:
        """
        Draws a rectangular box on the canvas with the given coordinates, color, and optional fill.

        Args:
            box: The box whose border we want to draw.
            color: Color of the box border.
        """

        # Left edge
        plt.plot([box.left, box.left], [box.top, box.bottom], color=color)
        # Right edge
        plt.plot([box.right, box.right], [box.top, box.bottom], color=color)
        # Top edge
        plt.plot([box.left, box.right], [box.top, box.top], color=color)
        # Bottom edge
        plt.plot([box.left, box.right], [box.bottom, box.bottom], color=color)

    def line(self, start: Vec2d, end: Vec2d, color: str):
        """
        Draws a line between two points on the canvas.

        Args:
            start (Vec2d): Starting point of the line.
            end (Vec2d): Ending point of the line.
            color: Color of the line.
        """

        plt.plot([start.x, end.x], [start.y, end.y], color=color)
