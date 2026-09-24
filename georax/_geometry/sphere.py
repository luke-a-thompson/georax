from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._charts import SphereChart
from .base import FrameCoords, LocalChart, Manifold, StateArray

__all__ = ["Sphere"]


class Sphere(Manifold["Sphere"]):
    """Unit vectors in R^n with the left rotation action A @ x.

    States have shape (..., n), with optional leading batch axes. Coordinates
    have shape (..., n * (n - 1) // 2), in row-major strictly lower-triangular
    order: (1, 0), (2, 0), (2, 1), (3, 0), ... . These are SO(n) generators,
    not independent tangent coordinates on the (n - 1)-dimensional sphere.
    """

    _chart_class = SphereChart
    n: int = eqx.field(static=True)

    def __init__(self, n: int) -> None:
        if not isinstance(n, int) or n < 2:
            raise ValueError("Sphere(n) requires an integer n >= 2.")
        self.n = n

    @property
    def state_shape(self) -> tuple[int]:
        return (self.n,)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return (self.n * (self.n - 1) // 2,)

    def check_state_shape(self, x: StateArray) -> None:
        if x.ndim == 0 or x.shape[-1:] != self.state_shape:
            raise ValueError(
                f"Sphere state must have shape (..., {self.n}); got {x.shape}."
            )

    def check_coordinate_shape(self, a: FrameCoords) -> None:
        size = self.coordinate_shape[0]
        if a.ndim == 0 or a.shape[-1:] != self.coordinate_shape:
            raise ValueError(
                f"Sphere coordinates must have shape (..., {size}); got {a.shape}."
            )

    def zero_coordinates(self, x: StateArray) -> FrameCoords:
        self.check_state_shape(x)
        return jnp.zeros(x.shape[:-1] + self.coordinate_shape, dtype=x.dtype)

    def _coords_to_alg(self, a: FrameCoords) -> Array:
        self.check_coordinate_shape(a)
        lower_i, lower_j = np.tril_indices(self.n, k=-1)
        omega = jnp.zeros(a.shape[:-1] + (self.n, self.n), dtype=a.dtype)
        omega = omega.at[..., lower_i, lower_j].set(a)
        return omega.at[..., lower_j, lower_i].set(-a)

    def _alg_to_coords(self, omega: Array) -> FrameCoords:
        if omega.shape[-2:] != (self.n, self.n):
            raise ValueError(
                f"Sphere rotation algebra must have shape (..., {self.n}, {self.n}); got {omega.shape}."
            )
        omega = 0.5 * (omega - jnp.swapaxes(omega, -1, -2))
        lower_i, lower_j = np.tril_indices(self.n, k=-1)
        return omega[..., lower_i, lower_j]

    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        """Lift a tangent v using v x^T - x v^T; radial components are discarded."""
        self.check_state_shape(x)
        self.check_state_shape(v)
        return self._alg_to_coords(
            v[..., :, None] * x[..., None, :] - x[..., :, None] * v[..., None, :]
        )

    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        self.check_state_shape(x)
        return (self._coords_to_alg(a) @ x[..., None])[..., 0]

    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        self.check_state_shape(x)
        lift_a, lift_b = self._coords_to_alg(a), self._coords_to_alg(b)
        # Fundamental fields of a left action have the negative algebra bracket.
        return self._alg_to_coords(lift_b @ lift_a - lift_a @ lift_b)

    def select_pullback_chart(self, required_order: int) -> LocalChart[Sphere]:
        """Use the exact Cayley coordinate map for pulled-back equations."""
        del required_order
        return self.select_chart(2)
