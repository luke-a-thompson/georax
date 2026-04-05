from __future__ import annotations

from typing import override

import equinox as eqx
from diffrax._custom_types import RealScalarLike
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jaxtyping import Array

from .base import LieGroup, LocalFlow, flow_order


class CayleyFlow(LocalFlow):
    order: flow_order = 2
    inverse_order: flow_order = 2

    def forward(self, x: Array, a: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(a, dtype=x.dtype)
        ident = jnp.eye(geometry.n, dtype=x.dtype)
        q = jnp.linalg.solve(ident - 0.5 * omega, ident + 0.5 * omega)
        return x @ q

    def d_inverse(self, x: Array, y: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(x)
        eta = geometry._coords_to_alg(y)
        ident = jnp.eye(geometry.n, dtype=eta.dtype)
        corrected = (ident - 0.5 * omega) @ eta @ (ident + 0.5 * omega)
        return geometry._alg_to_coords(corrected)


class RodriguesFlow(LocalFlow):
    order: flow_order = "exact"
    inverse_order: flow_order = "exact"

    def forward(self, x: Array, a: Array, geometry: SO) -> Array:
        if geometry.n != 3:
            raise ValueError("Rodrigues formula is only implemented for SO(3).")

        omega = geometry._coords_to_alg(a, dtype=x.dtype)
        theta_sq = 0.5 * jnp.sum(omega * omega)
        theta = jnp.sqrt(theta_sq)
        sin_over_theta = jnp.where(
            theta_sq > 1e-16,
            jnp.sin(theta) / theta,
            1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        )
        one_minus_cos_over_theta_sq = jnp.where(
            theta_sq > 1e-16,
            (1.0 - jnp.cos(theta)) / theta_sq,
            0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        )
        ident = jnp.eye(geometry.n, dtype=x.dtype)
        q = (
            ident
            + sin_over_theta * omega
            + one_minus_cos_over_theta_sq * (omega @ omega)
        )
        return x @ q

    def d_inverse(self, x: Array, y: Array, geometry: SO) -> Array:
        raise NotImplementedError("Rodrigues differential inverse is not implemented.")


class ExpFlow(LocalFlow):
    order: flow_order = "exact"
    inverse_order: flow_order = "exact"

    def forward(self, x: Array, a: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(a, dtype=x.dtype)
        return x @ jsp_linalg.expm(omega)

    def d_inverse(self, x: Array, y: Array, geometry: SO) -> Array:
        raise NotImplementedError(
            "Exponential differential inverse is not implemented."
        )


class SO(LieGroup):
    """SO(n) with a left-invariant frame and Cayley retraction."""

    n: int = eqx.field(static=True)
    _upper_i: Array
    _upper_j: Array
    _basis: Array

    def __init__(self, n: int, *, flow: LocalFlow | None = None):
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
        object.__setattr__(self, "flow", CayleyFlow() if flow is None else flow)

    @property
    def lie_algebra_dimension(self) -> int:
        return int(self._upper_i.size)

    def _coords_to_alg(self, a: Array, *, dtype=None) -> Array:
        coeffs = jnp.asarray(a, dtype=dtype)
        omega = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        omega = omega.at[self._upper_i, self._upper_j].set(coeffs)
        omega = omega.at[self._upper_j, self._upper_i].set(-coeffs)
        return omega

    def _alg_to_coords(self, omega: Array) -> Array:
        omega = 0.5 * (omega - omega.T)
        return omega[self._upper_i, self._upper_j]

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
        omega = self._coords_to_alg(a, dtype=x.dtype)
        return x @ omega

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        return self.frozen_flow(x, self.to_frame(x, v))

    @override
    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        if self.flow is None:
            raise NotImplementedError("No local flow is attached to this geometry.")
        return self.flow.d_inverse(a, b, self)

    @override
    def select_flow_method(self, required_order: RealScalarLike) -> LocalFlow:
        match required_order:
            case order if order <= 2:
                flow: LocalFlow = CayleyFlow()
            case _:
                flow = RodriguesFlow() if self.n == 3 else ExpFlow()

        object.__setattr__(self, "flow", flow)
        return flow
