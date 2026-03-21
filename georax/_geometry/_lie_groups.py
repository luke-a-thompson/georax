from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .base import GeometricOps


class SO(GeometricOps):
    """SO(n) with a left-invariant frame and Cayley retraction."""

    n: int = eqx.field(static=True)
    _upper_i: Array
    _upper_j: Array
    _basis: Array

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

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "_upper_i", jnp.asarray(upper_i))
        object.__setattr__(self, "_upper_j", jnp.asarray(upper_j))
        object.__setattr__(self, "_basis", jnp.asarray(basis))

    @property
    def lie_algebra_dimension(self) -> int:
        return int(self._upper_i.size)

    @override
    def frame(self, x: Array) -> Array:
        basis = jnp.asarray(self._basis, dtype=x.dtype)
        return jnp.einsum("ab,bck->ack", x, basis)

    @override
    def to_frame(self, x: Array, v: Array) -> Array:
        omega = x.T @ v
        return omega[self._upper_i, self._upper_j]

    @override
    def from_frame(self, x: Array, a: Array) -> Array:
        coeffs = jnp.asarray(a)
        omega = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        omega = omega.at[self._upper_i, self._upper_j].set(coeffs)
        omega = omega.at[self._upper_j, self._upper_i].set(-coeffs)
        return x @ omega

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        omega = x.T @ v
        ident = jnp.eye(self.n, dtype=x.dtype)
        q = jnp.linalg.solve(ident - 0.5 * omega, ident + 0.5 * omega)
        return x @ q


SO3 = SO(3)
