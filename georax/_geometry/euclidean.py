from __future__ import annotations

from diffrax._custom_types import RealScalarLike
from jaxtyping import Array

from .base import LocalChart, Manifold, chart_order


class EuclideanChart(LocalChart["Euclidean"]):
    order: RealScalarLike = 12
    inverse_order: RealScalarLike = 12

    def apply(self, x: Array, a: Array, geometry: Euclidean) -> Array:
        del geometry
        return x + a

    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: Euclidean
    ) -> Array:
        del x, a, geometry
        return b


class Euclidean(Manifold["Euclidean"]):
    """R^n with the standard frame, addition retraction, and exact chart."""

    def select_chart(self, required_order: RealScalarLike) -> LocalChart[Euclidean]:
        del required_order
        chart: LocalChart[Euclidean] = EuclideanChart()
        object.__setattr__(self, "chart", chart)
        return chart
