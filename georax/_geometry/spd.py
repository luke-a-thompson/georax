from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._charts import (
    SPDChart,
    _sym,
)
from .base import Manifold

__all__ = ["SPD"]


class SPD(Manifold["SPD"]):
    """SPD(n) with scaled-``vech`` symmetric-lift coordinates.

    Coefficients are represented in the symmetric matrix algebra. The induced
    infinitesimal action is

        E_A(x) = A x + x A,

    and the local chart is the exact congruence action

        Phi_A(x) = exp(A) x exp(A).
    """

    _chart_class = SPDChart
    n: int = eqx.field(static=True)
    _diag_i: Array
    _upper_i: Array
    _upper_j: Array
    _basis: Array

    def __init__(self, n: int):
        n = int(n)
        if n < 1:
            raise ValueError("SPD(n) requires n >= 1.")

        diag_i = np.arange(n)
        upper_i, upper_j = np.triu_indices(n, k=1)
        d = n * (n + 1) // 2
        basis = np.zeros((n, n, d), dtype=float)

        basis[diag_i, diag_i, diag_i] = 1.0
        if upper_i.size:
            k = np.arange(n, d)
            scale = 1.0 / np.sqrt(2.0)
            basis[upper_i, upper_j, k] = scale
            basis[upper_j, upper_i, k] = scale

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "_diag_i", jnp.asarray(diag_i))
        object.__setattr__(self, "_upper_i", jnp.asarray(upper_i))
        object.__setattr__(self, "_upper_j", jnp.asarray(upper_j))
        object.__setattr__(self, "_basis", jnp.asarray(basis))

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return (self.n * (self.n + 1) // 2,)

    def _coords_to_sym(self, a: Array) -> Array:
        self.check_coordinate_shape(a)
        coeffs = jnp.asarray(a)
        tangent = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        tangent = tangent.at[self._diag_i, self._diag_i].set(coeffs[: self.n])
        if self.coordinate_shape[0] > self.n:
            sqrt_two = jnp.asarray(np.sqrt(2.0), dtype=coeffs.dtype)
            off_diag = coeffs[self.n :] / sqrt_two
            tangent = tangent.at[self._upper_i, self._upper_j].set(off_diag)
            tangent = tangent.at[self._upper_j, self._upper_i].set(off_diag)
        return tangent

    def _sym_to_coords(self, tangent: Array) -> Array:
        if tangent.shape != self.state_shape:
            raise ValueError(
                f"{type(self).__name__} symmetric matrix must have shape {self.state_shape}; got {tangent.shape}."
            )
        tangent = _sym(jnp.asarray(tangent))
        diag = tangent[self._diag_i, self._diag_i]
        sqrt_two = jnp.asarray(np.sqrt(2.0), dtype=tangent.dtype)
        off_diag = sqrt_two * tangent[self._upper_i, self._upper_j]
        return jnp.concatenate((diag, off_diag))

    def trivialise(self, x: Array, v: Array) -> Array:
        self.check_state_shape(x)
        self.check_state_shape(v)
        eigvals, eigvecs = jnp.linalg.eigh(_sym(x))
        local_v = eigvecs.T @ _sym(v) @ eigvecs
        local_a = local_v / (eigvals[:, None] + eigvals[None, :])
        return self._sym_to_coords(eigvecs @ local_a @ eigvecs.T)

    def detrivialise(self, x: Array, a: Array) -> Array:
        self.check_state_shape(x)
        lift = self._coords_to_sym(a)
        return lift @ x + x @ lift

    def frame_bracket(self, a: Array, b: Array) -> Array:
        del a, b
        raise NotImplementedError(
            "SPD frame coordinates are not closed under commutator."
        )
