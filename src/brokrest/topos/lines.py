# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import typing

import torch
from bokeh import plotting

from .probs import Importance
from .rects import Box, Segment
from .topos import Topo

if typing.TYPE_CHECKING:
    from .polygons import Polygon

__all__ = ["Line", "Point"]


class Point(Topo):
    "A collection of points."

    x: torch.Tensor
    "The x element."

    y: torch.Tensor
    "The y element."

    @typing.no_type_check
    def cross_eq_1d(self, points: typing.Self) -> torch.Tensor:
        """
        Find self == points, using cross product.
        The result would be in a `[len(self), len(points)]` boolean matrix.
        """

        if not (self.ndim == points.ndim == 1):
            raise ValueError(
                f"Only supports when both {self.ndim=} = {points.ndim=} = 1."
            )
        eq: Point = self[:, None] == points[None, :]
        result = eq.x & eq.y
        assert result.shape == (len(self), len(points)), result.shape
        return result

    def is_vertex_of(self, polygon: "Polygon") -> torch.Tensor:
        "Find if `self` is a polygon vertex of `polygon`. Return a boolean tensor."
        result = self.cross_eq_1d(polygon.vertices).any(dim=1)
        assert result.ndim == 1
        assert result.shape == (len(self),)
        return result

    @typing.override
    def _outer(self) -> Box:
        return Box(x_0=self.x, x_1=self.x, y_0=self.y, y_1=self.y)

    def unit(self):
        return self / self.length

    @property
    def length(self) -> float:
        return (self.x**2 + self.y**2).sum().item() ** 0.5

    @typing.override
    def plot(self, figure: plotting.figure, /) -> None:
        _ = figure.scatter(x=self.x.numpy(), y=self.y.numpy(), color="red")


def mean_squared_error(x: torch.Tensor):
    return (x**2).mean()


class Line(Topo):
    """
    A set of lines. Represented as `ax + by + c = 0` (standard form).
    """

    a: torch.Tensor
    "The x coefficient."

    b: torch.Tensor
    "The y coefficient."

    c: torch.Tensor
    "The constant term."

    @typing.override
    def _setup_batch_size(self) -> torch.Size:
        sizes = {s for s in [self.a.shape, self.b.shape, self.c.shape] if len(s)}

        if len(sizes) == 0:
            return NotImplemented

        if len(sizes) != 1:
            raise ValueError(f"Too many sizes: {sizes=}.")

        target_size = list(sizes)[0]

        def _cast_to_target_size(item: torch.Tensor):
            if item.shape == target_size:
                return item

            assert item.shape == ()
            return item * torch.ones(target_size)

        self.a = _cast_to_target_size(self.a)
        self.b = _cast_to_target_size(self.b)
        self.c = _cast_to_target_size(self.c)

        self.auto_batch_size_()
        return self.batch_size

    @property
    def slope(self) -> torch.Tensor:
        return -self.a / self.b

    @property
    def x_intercept(self) -> torch.Tensor:
        return self.b

    @property
    def y_intercept(self) -> torch.Tensor:
        return self.b

    def solve_y(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns y = mx + b as a self.ndim + 1 matrix `R`. `R_ij = m_i x_j + b_i.`
        """

        return self.slope[..., None] * x[None, ...] + self.y_intercept[..., None]

    def subs(self, points: Point) -> torch.Tensor:
        """
        Returns ax + by + c.
        """

        a = self.a[..., None]
        b = self.b[..., None]
        c = self.c[..., None]
        x = points.x[None, ...]
        y = points.y[None, ...]

        return a * x + b * y + c

    def dist(self, points: Point) -> torch.Tensor:
        """
        Compute the distance of each points to a line.
        """

        dist_mat = self.flatten()._dist_prod_flat(points.flatten())
        assert dist_mat.ndim == 2, dist_mat.shape

        # Cast it to [*self.shape, *self.points] dims.
        result = dist_mat
        result = _unflatten(result, -1, points.shape)
        result = _unflatten(result, 0, self.shape)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (*self.shape, *points.shape)
        return result

    def dist_loss_score(
        self,
        points: Point,
        resample: Importance = NotImplemented,
    ) -> torch.Tensor:
        """
        Convert the distance into a loss (larger = farther).
        """

        dist = self.dist(points)

        if resample is not NotImplemented:
            dist = resample(dist)

        score = mean_squared_error(dist)

        assert score.ndim == 0, score.shape
        assert (score >= 0).item(), score
        return score

    def _dist_prod_flat(self, points: Point) -> torch.Tensor:
        """
        The distance product. `self` and `points` are both 1D (`flatten()`-ed).

        Note that this uses + and - values to show the sides at which the points are.
        """

        if self.ndim != 1 or points.ndim != 1:
            raise ValueError("Only supports 1d lines and points.")

        # The broadcasted dimensions would be [num_lines, num_points].
        ss: typing.Self = self[:, None]
        ps: Point = points[None, :]

        return ss.a * ps.x + ss.b * ps.y + ss.c

    @typing.override
    def _outer(self) -> "Box":
        return NotImplemented

    plot = NotImplemented

    @classmethod
    def standard(cls, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> typing.Self:
        "Create a line in the `ax + by + c = 0` form."

        return cls(a=a, b=b, c=c)

    @classmethod
    def intercept(cls, a: torch.Tensor, b: torch.Tensor) -> typing.Self:
        "Create a line in the `x/a + y/b = 1` form."

        return cls(a=1 / a, b=1 / b, c=torch.tensor(-1))

    @classmethod
    def slope_intercept(cls, m: torch.Tensor, b: torch.Tensor) -> typing.Self:
        "Create a line in the `y = mx + b` form."

        return cls(a=m, b=torch.tensor(-1), c=b)

    @classmethod
    def from_segment(cls, segment: Segment) -> typing.Self:
        "Extend the segment into a line."

        slope = segment.slope
        end = segment.end
        return cls.slope_intercept(slope, end.y - slope * end.x)


def _unflatten(item: torch.Tensor, dim: int, sizes: tuple[int, ...]) -> torch.Tensor:
    if not sizes:
        return item.unsqueeze(dim)

    else:
        return item.unflatten(dim, sizes)
