from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._charts import SOChart
from .base import FrameCoords, LocalChart, Manifold, StateArray

__all__ = ["SO"]


class SO(Manifold["SO"]):
    """SO(n) with a left-invariant frame and order-dependent local chart."""

    _chart_class = SOChart
    n: int = eqx.field(static=True)

    def __init__(self, n: int) -> None:
        n = int(n)
        if n < 2:
            raise ValueError("SO(n) requires n >= 2.")

        object.__setattr__(self, "n", n)

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return (self.n * (self.n - 1) // 2,)

    def _coords_to_alg(self, a: FrameCoords) -> Array:
        self.check_coordinate_shape(a)
        coeffs = jnp.asarray(a)
        omega = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        upper_i, upper_j = np.triu_indices(self.n, k=1)
        omega = omega.at[upper_i, upper_j].set(coeffs)
        omega = omega.at[upper_j, upper_i].set(-coeffs)
        return omega

    def _alg_to_coords(self, omega: Array) -> FrameCoords:
        if omega.shape != self.state_shape:
            raise ValueError(
                f"{type(self).__name__} Lie algebra matrix must have shape {self.state_shape}; got {omega.shape}."
            )
        omega = 0.5 * (omega - omega.T)
        return omega[np.triu_indices(self.n, k=1)]

    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        self.check_state_shape(x)
        self.check_state_shape(v)
        return self._alg_to_coords(x.T @ v)

    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        self.check_state_shape(x)
        return x @ self._coords_to_alg(a)

    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        del x
        lift_a = self._coords_to_alg(a)
        lift_b = self._coords_to_alg(b)
        return self._alg_to_coords(lift_a @ lift_b - lift_b @ lift_a)

    def select_pullback_chart(self, required_order: int) -> LocalChart[SO]:
        """Use Cayley as an exact coordinate map for pulled-back equations."""
        del required_order
        return self.select_chart(2)
