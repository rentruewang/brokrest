# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of shapes that can be represented as a 4 tuple."

import abc
import dataclasses as dcls
import typing

import numpy as np
import shapely
from bokeh import plotting

from brokrest.plotting import ViewPort
from brokrest.typing import FloatArray, IntArray

from .topos import Topo

if typing.TYPE_CHECKING:
    from .lines import Point

__all__ = ["Rect", "Box", "Segment"]


@dcls.dataclass
class Rect(Topo, abc.ABC):
    """
    A tuple with 4 values.

    If `batch_size` is not 1, all `x_0`, `x_1`, `y_0`, `y_1` need to be the same shape.
    """

    x_0: FloatArray
    "The left side."

    x_1: FloatArray
    "The right side."

    y_0: FloatArray
    "The top side."

    y_1: FloatArray
    "The bottom side."


class Box(Rect):
    """
    A box with 4 sides.
    """

    @typing.override
    def _checks(self) -> None:
        super()._checks()

        if not np.all(self.width >= 0):
            raise ValueError("Box should not have negative width.")

        if not np.all(self.height >= 0):
            raise ValueError("Box should not have negative height.")

    @property
    def bottom(self) -> FloatArray:
        "The bottom of the box. Alias of `y_0`."
        return self.y_0

    @property
    def top(self) -> FloatArray:
        "The top of the box. Alias of `y_1`."
        return self.y_1

    @property
    def left(self) -> FloatArray:
        "The left of the box. Alias of `x_0`."
        return self.x_0

    @property
    def right(self) -> FloatArray:
        "The right of the box. Alias of `x_1`."
        return self.x_1

    @property
    def width(self) -> FloatArray:
        "The width of the `Box`es."
        return self.x_1 - self.x_0

    @property
    def height(self) -> FloatArray:
        "The height of the `Box`es."
        return self.y_1 - self.y_0

    @property
    def area(self) -> FloatArray:
        "The area of the `Box`es."
        return self.width * self.height

    @typing.override
    def _outer(self):
        return self

    def boundary(self):
        return shapely.box(
            xmin=self.left, xmax=self.right, ymin=self.bottom, ymax=self.top
        )

    def convex_hull(self):
        return shapely.convex_hull(self.boundary())

    @typing.override
    def plot(self, figure: plotting.figure) -> None:
        _ = figure.rect(x=self.x_0, y=self.y_0, width=self.width, height=self.height)

    def visible(self, window: ViewPort) -> FloatArray:
        """
        Return a boolean tensor, of whether `self` is visible in the view box or not.

        Args:
            window: The view port to determine.

        Returns:
            A boolean tensor the same length as `self`.
        """

        horiz = _segment_visible(
            start=self.left,
            end=self.right,
            x=window.left,
            y=window.right,
        )
        verti = _segment_visible(
            start=self.bottom,
            end=self.top,
            x=window.bottom,
            y=window.top,
        )

        # Both horizontally and vertically visible.
        return horiz & verti


def _segment_visible(
    start: FloatArray, end: FloatArray, x: float, y: float
) -> FloatArray:
    """
    Try to see if segment [start, end] is visible in viewport [x, y], vectorized.
    """

    result_shape = start.shape
    assert end.shape == start.shape

    def is_ordered(*ordered: FloatArray):
        "The tensors are ordered."

        answer = np.ones(result_shape).astype("bool")
        for smaller, larger in zip(ordered[:-1], ordered[1:]):
            answer &= smaller <= larger
        return answer

    x_tensor = np.array(x).astype("float32")
    y_tensor = np.array(y).astype("float32")

    # Let's laid it out on an axis.
    # the line is visible with one of the conditions:
    ans = np.zeros(result_shape).astype("bool")

    # start - x - end - y
    ans |= is_ordered(start, x_tensor, end, y_tensor)

    # start - x - y - end
    ans |= is_ordered(start, x_tensor, y_tensor, end)

    # x - start - y - end
    ans |= is_ordered(x_tensor, start, y_tensor, end)

    # x - start - end - y
    ans |= is_ordered(x_tensor, start, end, y_tensor)

    return ans


class Segment(Rect):
    "Segments on a 2D plane."

    @typing.override
    def ordering(self) -> IntArray:
        return np.argsort(self.x_0)

    @property
    def dx(self) -> FloatArray:
        return self.x_1 - self.x_0

    @property
    def dy(self) -> FloatArray:
        return self.y_1 - self.y_0

    @property
    def slope(self) -> FloatArray:
        return self.dy / self.dx

    @property
    def angle(self):
        return (self.dx + self.dy * 1j).angle()

    def flip(self):
        return type(self)(x_0=self.x_1, y_0=self.y_1, x_1=self.x_0, y_1=self.y_0)

    def face_right(self) -> typing.Self:
        self = self.flatten()
        needs_flipping = self.x_0 > self.x_1
        return td.cat([self[~needs_flipping], self[needs_flipping].flip()])

    @property
    def is_ltr_1d(self):
        "Is left to right."

        if self.ndim != 1:
            raise RuntimeError(f"Left to right only makes sense for 1D. {self.ndim=}.")

        return (self.x_0[1:] >= self.x_0[:-1]).all()

    def merge_similar_mono(self, radian: float = 0.1) -> typing.Self:
        """
        Merge consecutive segments with angle diff under `radian`.
        """

        # No need to update.
        if not self.is_ltr_1d:
            raise ValueError(f"{self} is not left to right.")

        angles = self.angle
        shifted = angles.roll(-1, 0)
        merge_at = (shifted - angles).abs() <= radian

        results: list[Segment] = []
        to_merge: list[Segment] = []
        for segment, do_merge in zip(self, merge_at):
            if do_merge:
                to_merge.append(segment)

            elif to_merge:
                assert segment.x_0 == to_merge[-1].x_1
                assert segment.y_0 == to_merge[-1].y_1
                results.append(
                    type(self)(
                        x_0=to_merge[0].x_0,
                        y_0=to_merge[0].y_0,
                        x_1=segment.x_1,
                        y_1=segment.y_1,
                    )
                )
                to_merge.clear()

            else:
                results.append(segment)

        return td.stack(results)

    @property
    def start(self):
        "The starting point of a segment."

        from .lines import Point

        return Point(x=self.x_0, y=self.y_0)

    @property
    def end(self):
        "The ending point of a segment."

        from .lines import Point

        return Point(x=self.x_1, y=self.y_1)

    @property
    def left(self):
        "The `min(x)`."

        return torch.minimum(self.x_0, self.x_1)

    @property
    def right(self):
        "The `max(x)`."

        return torch.maximum(self.x_0, self.x_1)

    @property
    def bottom(self):
        "The `min(y)`."
        return torch.minimum(self.y_0, self.y_1)

    @property
    def top(self):
        "The `max(y)`."
        return torch.maximum(self.y_0, self.y_1)

    def points(self):
        """
        Flatten the segment, then convert to points.
        """

        from .lines import Point

        # Shape: [*self.shapes, 2]
        start, end = self.start.tensor(), self.end.tensor()
        unique = torch.stack([start, end]).view(2, -1).unique(dim=0)
        return Point(unique[0], unique[1])

    def line(self):
        from .lines import Line

        return Line.from_segment(self)

    @typing.override
    def _outer(self):
        return Box(x_0=self.left, x_1=self.right, y_0=self.bottom, y_1=self.top)

    @typing.override
    def plot(self, figure: plotting.figure) -> None:
        _ = figure.segment(
            x0=self.x_0.numpy(),
            x1=self.x_1.numpy(),
            y0=self.y_0.numpy(),
            y1=self.y_1.numpy(),
        )

    @classmethod
    def from_start_end(cls, start: "Point", end: "Point") -> typing.Self:
        return cls(x_0=start.x, y_0=start.y, x_1=end.x, y_1=end.y)

    @classmethod
    def from_points(cls, points: "Point") -> typing.Self:
        return cls.from_start_end(points[1:], points[:-1])
