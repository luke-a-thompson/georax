from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from .base import FrameCoords, LocalChart, Manifold, StateArray


class EuclideanChart(LocalChart["Euclidean"]):
    def apply(self, x: Array, a: FrameCoords, geometry: Euclidean) -> Array:
        del geometry
        return x + a

    def inverse_differential(
        self, x: Array, a: FrameCoords, b: FrameCoords, geometry: Euclidean
    ) -> FrameCoords:
        del x, a, geometry
        return b


class Euclidean(Manifold["Euclidean"]):
    """R^n with the standard frame and addition retraction."""

    _chart_class = EuclideanChart

    @property
    def state_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def coordinate_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def check_state_shape(self, x: StateArray) -> None:
        del x

    def check_coordinate_shape(self, a: FrameCoords) -> None:
        del a

    def zero_coordinates(self, x: StateArray) -> FrameCoords:
        return jnp.zeros_like(x)

    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        del x
        return v

    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        del x
        return a

    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        del x
        return jnp.zeros_like(a + b)
