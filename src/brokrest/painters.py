# Copyright (c) The BrokRest Authors - All Rights Reserved

"The canvas that handles plotting."

import dataclasses as dcls

import seaborn as sns
from matplotlib import pyplot as plt

from .geo import Vec2d


@dcls.dataclass(frozen=True)
class Canvas:
    left: int
    "The left most boundary."

    right: int
    "The right most boundary."

    def __post_init__(self):
        sns.set_theme()
        plt.clf()

    def fill(
        self,
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        color: str,
    ) -> None:
        """
        Draws a rectangular box on the canvas with the given coordinates, color, and optional fill.

        Args:
            left: Left coordinate of the box.
            right: Right coordinate of the box.
            top: Top coordinate of the box.
            bottom: Bottom coordinate of the box.
            color: Color of the box border.
            fill: Whether to fill the box with color. Default is False.
        """

        plt.fill(
            [left, left, right, right],
            [top, bottom, bottom, top],
            color=color,
        )

    def border(
        self,
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        color: str,
    ) -> None:
        """
        Draws a rectangular box on the canvas with the given coordinates, color, and optional fill.

        Args:
            left: Left coordinate of the box.
            right: Right coordinate of the box.
            top: Top coordinate of the box.
            bottom: Bottom coordinate of the box.
            color: Color of the box border.
        """

        plt.plot([left, left], [top, bottom], color=color)  # Left edge
        plt.plot([right, right], [top, bottom], color=color)  # Right edge
        plt.plot([left, right], [top, top], color=color)  # Top edge
        plt.plot([left, right], [bottom, bottom], color=color)  # Bottom edge

    def line(self, start: Vec2d, end: Vec2d, color: str):
        """
        Draws a line between two points on the canvas.

        Args:
            start (Vec2d): Starting point of the line.
            end (Vec2d): Ending point of the line.
            color: Color of the line.
        """

        plt.plot([start.x, end.x], [start.y, end.y], color=color)
