from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from .base import FrameCoords, LocalChart, Manifold, StateMatrix


class EuclideanChart(LocalChart["Euclidean"]):
    order: int

    def __init__(self, order: int):
        del order
        object.__setattr__(self, "order", 12)

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
    def state_shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def coordinate_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def check_state_shape(self, x: StateMatrix) -> None:
        pass

    def check_coordinate_shape(self, a: FrameCoords) -> None:
        pass

    def trivialise(self, x: StateMatrix, v: StateMatrix) -> FrameCoords:
        del x
        return v

    def detrivialise(self, x: StateMatrix, a: FrameCoords) -> StateMatrix:
        del x
        return a

    def frame_bracket(
        self, x: StateMatrix, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        del x
        return jnp.zeros_like(a + b)
