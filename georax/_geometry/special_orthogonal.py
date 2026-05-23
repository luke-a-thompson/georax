from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._charts import SOChart
from .base import FrameCoords, Manifold, StateMatrix

__all__ = ["SO"]


class SO(Manifold["SO"]):
    """SO(n) with a left-invariant frame and Cayley retraction."""

    _chart_class = SOChart
    n: int = eqx.field(static=True)
    _upper_i: Array
    _upper_j: Array
    _basis: Array
    _structure_constants: Array

    def __init__(self, n: int):
        n = int(n)
        if n < 2:
            raise ValueError("SO(n) requires n >= 2.")

        upper_i, upper_j = np.triu_indices(n, k=1)
        d = upper_i.size
        basis = np.zeros((n, n, d), dtype=float)
        k = np.arange(d)
        basis[upper_i, upper_j, k] = 1.0
        basis[upper_j, upper_i, k] = -1.0

        structure = np.zeros((d, d, d), dtype=basis.dtype)
        for i in range(d):
            for j in range(d):
                commutator = basis[:, :, i] @ basis[:, :, j]
                commutator -= basis[:, :, j] @ basis[:, :, i]
                structure[:, i, j] = commutator[upper_i, upper_j]

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "_upper_i", jnp.asarray(upper_i))
        object.__setattr__(self, "_upper_j", jnp.asarray(upper_j))
        object.__setattr__(self, "_basis", jnp.asarray(basis))
        object.__setattr__(self, "_structure_constants", jnp.asarray(structure))

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return (int(self._upper_i.size),)

    def _coords_to_alg(self, a: FrameCoords) -> Array:
        self.check_coordinate_shape(a)
        coeffs = jnp.asarray(a)
        omega = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        omega = omega.at[self._upper_i, self._upper_j].set(coeffs)
        omega = omega.at[self._upper_j, self._upper_i].set(-coeffs)
        return omega

    def _alg_to_coords(self, omega: Array) -> FrameCoords:
        if omega.shape != self.state_shape:
            raise ValueError(
                f"{type(self).__name__} Lie algebra matrix must have shape {self.state_shape}; got {omega.shape}."
            )
        omega = 0.5 * (omega - omega.T)
        return omega[self._upper_i, self._upper_j]

    def trivialise(self, x: StateMatrix, v: StateMatrix) -> FrameCoords:
        self.check_state_shape(x)
        self.check_state_shape(v)
        return self._alg_to_coords(x.T @ v)

    def detrivialise(self, x: StateMatrix, a: FrameCoords) -> StateMatrix:
        self.check_state_shape(x)
        return x @ self._coords_to_alg(a)

    def frame_bracket(
        self, x: StateMatrix, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        del x
        return jnp.einsum("kij,i,j->k", self._structure_constants, a, b)
