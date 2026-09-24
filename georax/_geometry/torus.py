from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .base import FrameCoords, Manifold, StateArray
from .euclidean import EuclideanChart

__all__ = ["Torus"]


class TorusChart(EuclideanChart["Torus"]):
    """Exact addition modulo 2 pi, at every requested chart order."""

    def apply(self, x: Array, a: FrameCoords, geometry: Torus) -> Array:
        del geometry
        return (x + a + jnp.pi) % (2 * jnp.pi) - jnp.pi


class Torus(Manifold["Torus"]):
    """The d-dimensional torus in angular coordinates in [-pi, pi)."""

    _chart_class = TorusChart
    d: int = eqx.field(static=True)

    def __init__(self, d: int) -> None:
        if not isinstance(d, int) or d < 1:
            raise ValueError("Torus(d) requires a positive integer d.")
        self.d = d

    @property
    def state_shape(self) -> tuple[int]:
        return (self.d,)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return self.state_shape

    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        self.check_state_shape(x)
        self.check_state_shape(v)
        return v

    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        self.check_state_shape(x)
        self.check_coordinate_shape(a)
        return a

    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        self.check_state_shape(x)
        self.check_coordinate_shape(a)
        self.check_coordinate_shape(b)
        return jnp.zeros_like(a + b)
