# Copyright (c) The BrokRest Authors - All Rights Reserved

"A set of linear equations."

import typing
from collections import abc as cabc

import torch
from bokeh import plotting

from brokrest.tds import tensorclass

from .rects import Box
from .topos import Topo

__all__ = ["Line", "Point"]


@tensorclass
class Point(Topo):
    "A collection of points."

    x: torch.Tensor
    "The x element."

    y: torch.Tensor
    "The y element."

    @typing.override
    def _outer(self) -> Box:
        return Box(x_0=self.x, x_1=self.x, y_0=self.y, y_1=self.y)

    def unit(self) -> typing.Self:
        return self / self.length

    @property
    def length(self) -> float:
        return (self.x**2 + self.y**2).sum().item() ** 0.5

    @typing.override
    def _draw(self, figure: plotting.figure, /) -> None:
        _ = figure.scatter(
            x=self.x.numpy(),
            y=self.y.numpy(),
        )


def _mean_squared_error(x: torch.Tensor):
    return (x**2).mean()


@tensorclass
class Line(Topo):
    """
    A set of lines. Represented as `y = mx + b` (slope intercept form).
    """

    m: torch.Tensor
    "The slope of the line."

    b: torch.Tensor
    "The bias of the line."

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns mx + b as a self.ndim + 1 matrix `R`. `R_ij = m_i x_j + b_i.`
        """

        return self.m[..., None] * x[None, ...] + self.b[..., None]

    def subs(self, points: Point) -> torch.Tensor:
        """
        Returns mx + b as a self.ndim + 1 matrix `R`.
        `R_ij = m_i x_j + b_i - y_j.`
        """

        return self.apply(points.x) - points.y[None, ...]

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
        dist_loss: cabc.Callable[[torch.Tensor], torch.Tensor] = _mean_squared_error,
    ) -> torch.Tensor:
        """
        Convert the distance into a loss (larger = farther).
        """

        dist = self.dist(points)
        score = dist_loss(dist)
        assert score.ndim == 0
        assert score.positive().item(), score
        return score

    def _dist_prod_flat(self, points: Point) -> torch.Tensor:
        """
        The distance product. `self` and `points` are both 1D (`flatten()`-ed).

        Note that this uses + and - values to show the sides at which the points are.
        """

        if self.ndim != 1 or points.ndim != 1:
            raise ValueError("Only supports 1d lines and points.")

        # The broadcasted dimensions would be [num_lines, num_points].
        ss = self[:, torch.newaxis]
        ps = points[torch.newaxis, :]

        return (ss.m * ps.x - ps.y + ss.b) / (ss.m**2 + 1)

    @typing.override
    def _outer(self) -> "Box":
        return NotImplemented

    @typing.override
    def _draw(self, canvas: plotting.figure) -> None:
        raise NotImplementedError

    @classmethod
    def standard(cls, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> typing.Self:
        "Create a line in the `ax + by + c = 0` form."

        # y = -a/b x - c/b
        return cls(m=-a / b, b=-c / b)

    @classmethod
    def intercept(cls, a: torch.Tensor, b: torch.Tensor) -> typing.Self:
        "Create a line in the `x/a + y/b = 1` form."

        # y = b - b/a x
        return cls(m=-b / a, b=b)

    @classmethod
    def slope_intercept(cls, m: torch.Tensor, b: torch.Tensor) -> typing.Self:
        "Create a line in the `y = mx + b` form."

        return cls(m=m, b=b)


def _unflatten(item: torch.Tensor, dim: int, sizes: tuple[int, ...]) -> torch.Tensor:
    if not sizes:
        return item.squeeze(dim)

    else:
        return item.unflatten(dim, sizes)
