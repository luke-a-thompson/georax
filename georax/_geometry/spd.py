from __future__ import annotations

from typing import override

import equinox as eqx
from diffrax._custom_types import RealScalarLike
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .base import LocalFlow, Manifold, flow_order


def _sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


def _matrix_exp_sym(s: Array) -> Array:
    s = _sym(s)
    evals, evecs = jnp.linalg.eigh(s)
    return (evecs * jnp.exp(evals)) @ evecs.T


class CongruenceExpFlow(LocalFlow):
    order: flow_order = "exact"
    inverse_order: flow_order = "exact"

    def forward(self, x: Array, a: Array, geometry: "SPD") -> Array:
        x = _sym(jnp.asarray(x))
        lift = geometry._coords_to_sym(a)
        g = _matrix_exp_sym(lift)
        return _sym(g @ x @ g)


class SPD(Manifold):
    """SPD(n) with scaled-``vech`` symmetric-lift coordinates.

    Coefficients are represented in the symmetric matrix algebra. The induced
    infinitesimal action is

        E_A(x) = A x + x A,

    and the frozen flow is the exact congruence action

        Phi_A(x) = exp(A) x exp(A).
    """

    n: int = eqx.field(static=True)
    _diag_i: Array
    _upper_i: Array
    _upper_j: Array
    _basis: Array

    def __init__(self, n: int, *, flow: LocalFlow | None = None):
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
        object.__setattr__(self, "flow", CongruenceExpFlow() if flow is None else flow)

    @property
    def dimension(self) -> int:
        return self.n * (self.n + 1) // 2

    def _coords_to_sym(self, a: Array) -> Array:
        coeffs = jnp.asarray(a)
        tangent = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        tangent = tangent.at[self._diag_i, self._diag_i].set(coeffs[: self.n])
        if self.dimension > self.n:
            sqrt_two = jnp.asarray(np.sqrt(2.0), dtype=coeffs.dtype)
            off_diag = coeffs[self.n :] / sqrt_two
            tangent = tangent.at[self._upper_i, self._upper_j].set(off_diag)
            tangent = tangent.at[self._upper_j, self._upper_i].set(off_diag)
        return tangent

    def _sym_to_coords(self, tangent: Array) -> Array:
        tangent = _sym(jnp.asarray(tangent))
        diag = tangent[self._diag_i, self._diag_i]
        sqrt_two = jnp.asarray(np.sqrt(2.0), dtype=tangent.dtype)
        off_diag = sqrt_two * tangent[self._upper_i, self._upper_j]
        return jnp.concatenate((diag, off_diag))

    @override
    def frame(self, x: Array) -> Array:
        x = _sym(jnp.asarray(x))
        basis = jnp.asarray(self._basis, dtype=x.dtype)
        left = jnp.einsum("abk,bc->ack", basis, x)
        right = jnp.einsum("ab,bck->ack", x, basis)
        return left + right

    @override
    def to_frame(self, x: Array, v: Array) -> Array:
        x = _sym(jnp.asarray(x))
        v = _sym(jnp.asarray(v))
        evals, evecs = jnp.linalg.eigh(x)
        v_hat = evecs.T @ v @ evecs
        denom = evals[:, None] + evals[None, :]
        lift_hat = v_hat / denom
        lift = _sym(evecs @ lift_hat @ evecs.T)
        return self._sym_to_coords(lift)

    @override
    def from_frame(self, x: Array, a: Array) -> Array:
        x = _sym(jnp.asarray(x))
        lift = self._coords_to_sym(a)
        return _sym(lift @ x + x @ lift)

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        return self.frozen_flow(x, self.to_frame(x, v))

    @override
    def select_flow_method(self, required_order: RealScalarLike) -> LocalFlow:
        del required_order
        flow: LocalFlow = CongruenceExpFlow()
        object.__setattr__(self, "flow", flow)
        return flow
