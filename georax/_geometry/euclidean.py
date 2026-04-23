from __future__ import annotations

from typing import override

from diffrax._custom_types import RealScalarLike
import jax.numpy as jnp
from jaxtyping import Array

from .base import LocalChart, Manifold, chart_order


class EuclideanChart(LocalChart):
    order: chart_order = "exact"
    inverse_order: chart_order = "exact"

    def apply(self, x: Array, a: Array, geometry: Euclidean) -> Array:
        return x + a

    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: Euclidean
    ) -> Array:
        del x, a, geometry
        return b


class Euclidean(Manifold):
    """R^n with the standard frame, addition retraction, and exact chart."""

    def __init__(self, *, chart: LocalChart | None = None):
        object.__setattr__(self, "chart", EuclideanChart() if chart is None else chart)

    @override
    def frame(self, x: Array) -> Array:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    @override
    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    @override
    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        return x + v

    @override
    def select_chart(self, required_order: RealScalarLike) -> LocalChart:
        del required_order
        chart: LocalChart = EuclideanChart()
        object.__setattr__(self, "chart", chart)
        return chart
