from __future__ import annotations

import jax.numpy as jnp
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array

from .base import LocalChart, Manifold


class EuclideanChart(LocalChart["Euclidean"]):
    order: RealScalarLike
    inverse_order: RealScalarLike

    def __init__(self, order: int):
        del order
        object.__setattr__(self, "order", 12)
        object.__setattr__(self, "inverse_order", 12)

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

    _chart_class = EuclideanChart

    @property
    def state_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def coordinate_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def check_state_shape(self, x: Array) -> None:
        pass

    def check_coordinate_shape(self, a: Array) -> None:
        pass

    def trivialise(self, x: Array, v: Array) -> Array:
        del x
        return v

    def detrivialise(self, x: Array, a: Array) -> Array:
        del x
        return a

    def frame_bracket(self, a: Array, b: Array) -> Array:
        return jnp.zeros_like(a + b)
