from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .base import FrameCoords, Geometry, LocalChart, Manifold, StateArray


class EuclideanChart(LocalChart[Geometry]):
    def apply(self, x: Array, a: FrameCoords, geometry: Geometry) -> Array:
        del geometry
        return x + a

    def inverse_differential(
        self, x: Array, a: FrameCoords, b: FrameCoords, geometry: Geometry
    ) -> FrameCoords:
        del x, a, geometry
        return b


class Euclidean(Manifold["Euclidean"]):
    """Euclidean arrays with an optional fixed shape and exact addition chart.

    ``Euclidean()`` accepts any array, while ``Euclidean(n)`` or
    ``Euclidean((m, n))`` fixes the shape for use as a product factor.
    """

    _chart_class = EuclideanChart
    shape: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(self, shape: int | tuple[int, ...] | None = None) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        if shape is not None:
            shape = tuple(shape)
            if any(not isinstance(size, int) or size < 1 for size in shape):
                raise ValueError(
                    "Euclidean shape dimensions must be positive integers."
                )
        self.shape = shape

    @property
    def state_shape(self) -> tuple[int, ...]:
        if self.shape is None:
            raise ValueError("Specify a Euclidean shape when using it in a Product.")
        return self.shape

    @property
    def coordinate_shape(self) -> tuple[int, ...]:
        return self.state_shape

    def check_state_shape(self, x: StateArray) -> None:
        if self.shape is not None:
            super().check_state_shape(x)

    def check_coordinate_shape(self, a: FrameCoords) -> None:
        if self.shape is not None:
            super().check_coordinate_shape(a)

    def zero_coordinates(self, x: StateArray) -> FrameCoords:
        self.check_state_shape(x)
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
