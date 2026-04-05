from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jaxtyping import Array

from .base import Manifold


def _sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


class SPDHomogeneous(Manifold):
    """SPD(n) as the homogeneous space GL(n)/O(n) via congruence action.

    Coefficients are represented in the full matrix Lie algebra gl(n). The induced
    frame is

        E_A(x) = A x + x A^T,

    and the exact frozen flow is

        Phi_A(x) = exp(A) x exp(A)^T.
    """

    n: int = eqx.field(static=True)
    _basis: Array
    _basis_t: Array

    def __init__(self, n: int):
        n = int(n)
        if n < 1:
            raise ValueError("SPDHomogeneous(n) requires n >= 1.")

        basis = np.zeros((n, n, n * n), dtype=float)
        for i in range(n):
            for j in range(n):
                basis[i, j, i * n + j] = 1.0

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "_basis", jnp.asarray(basis))
        object.__setattr__(self, "_basis_t", jnp.asarray(np.swapaxes(basis, 0, 1)))

    @property
    def lie_algebra_dimension(self) -> int:
        return self.n * self.n

    def _coords_to_alg(self, a: Array) -> Array:
        a = jnp.asarray(a)
        if a.shape == (self.n, self.n):
            return a
        return jnp.reshape(a, (self.n, self.n))

    def _alg_to_coords(self, a: Array) -> Array:
        return jnp.reshape(jnp.asarray(a), (self.n * self.n,))

    @override
    def frame(self, x: Array) -> Array:
        x = _sym(jnp.asarray(x))
        basis = jnp.asarray(self._basis, dtype=x.dtype)
        basis_t = jnp.asarray(self._basis_t, dtype=x.dtype)
        left = jnp.einsum("abk,bc->ack", basis, x)
        right = jnp.einsum("ab,bck->ack", x, basis_t)
        return left + right

    @override
    def to_frame(self, x: Array, v: Array) -> Array:
        x = _sym(jnp.asarray(x))
        v = _sym(jnp.asarray(v))
        alg = 0.5 * jnp.linalg.solve(x, v.T).T
        return self._alg_to_coords(alg)

    @override
    def from_frame(self, x: Array, a: Array) -> Array:
        x = _sym(jnp.asarray(x))
        alg = self._coords_to_alg(a)
        return _sym(alg @ x + x @ alg.T)

    def frozen_flow(self, x: Array, a: Array) -> Array:
        x = _sym(jnp.asarray(x))
        alg = self._coords_to_alg(a)
        g = jsp_linalg.expm(alg)
        return _sym(g @ x @ g.T)

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        return self.frozen_flow(x, self.to_frame(x, v))
