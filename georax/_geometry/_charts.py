from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jaxtyping import Array

from .base import LocalChart, chart_order

if TYPE_CHECKING:
    from .spd import SPD
    from .special_orthogonal import SO


def _pade_expm(a: Array, order: int) -> Array:
    degree = max(1, (int(order) + 1) // 2)
    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    powers = [ident]
    for _ in range(degree):
        powers.append(powers[-1] @ a)

    coeffs = [
        math.factorial(2 * degree - k)
        * math.factorial(degree)
        / (
            math.factorial(2 * degree)
            * math.factorial(k)
            * math.factorial(degree - k)
        )
        for k in range(degree + 1)
    ]
    numerator = jnp.zeros_like(a)
    denominator = jnp.zeros_like(a)
    for k, (coeff, power) in enumerate(zip(coeffs, powers, strict=True)):
        scaled = jnp.asarray(coeff, dtype=a.dtype) * power
        numerator = numerator + scaled
        denominator = denominator + ((-1) ** k) * scaled
    return jnp.linalg.solve(denominator, numerator)


def _sym(a: Array) -> Array:
    return 0.5 * (a + a.T)


def _matrix_exp_sym(s: Array) -> Array:
    return _sym(jsp_linalg.expm(_sym(s)))


# ---------------------------------------------------------------------------
# SO(n) charts
# ---------------------------------------------------------------------------


class CayleyChart(LocalChart):
    order: chart_order = 2
    inverse_order: chart_order = 2

    def apply(self, x: Array, a: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(a, dtype=x.dtype)
        ident = jnp.eye(geometry.n, dtype=x.dtype)
        q = jnp.linalg.solve(ident - 0.5 * omega, ident + 0.5 * omega)
        return x @ q

    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: SO
    ) -> Array:
        del x
        omega = geometry._coords_to_alg(a)
        eta = geometry._coords_to_alg(b)
        ident = jnp.eye(geometry.n, dtype=eta.dtype)
        corrected = (ident - 0.5 * omega) @ eta @ (ident + 0.5 * omega)
        return geometry._alg_to_coords(corrected)


class RodriguesChart(LocalChart):
    order: chart_order = "exact"
    inverse_order: chart_order = "exact"

    def apply(self, x: Array, a: Array, geometry: SO) -> Array:
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


class ExpChart(LocalChart):
    order: chart_order = "exact"
    inverse_order: chart_order = "exact"

    def apply(self, x: Array, a: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(a, dtype=x.dtype)
        return x @ jsp_linalg.expm(omega)


class PadeChart(LocalChart):
    order: chart_order
    inverse_order: chart_order

    def __init__(self, order: int):
        object.__setattr__(self, "order", int(order))
        object.__setattr__(self, "inverse_order", int(order))

    def apply(self, x: Array, a: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(a, dtype=x.dtype)
        return x @ _pade_expm(omega, self.order)


# ---------------------------------------------------------------------------
# SPD(n) charts
# ---------------------------------------------------------------------------


class CongruenceExpChart(LocalChart):
    order: chart_order = "exact"
    inverse_order: chart_order = "exact"

    def apply(self, x: Array, a: Array, geometry: SPD) -> Array:
        x = _sym(jnp.asarray(x))
        lift = geometry._coords_to_sym(a)
        g = _matrix_exp_sym(lift)
        return _sym(g @ x @ g)


class CongruencePadeChart(LocalChart):
    order: chart_order
    inverse_order: chart_order

    def __init__(self, order: int):
        object.__setattr__(self, "order", int(order))
        object.__setattr__(self, "inverse_order", int(order))

    def apply(self, x: Array, a: Array, geometry: SPD) -> Array:
        x = _sym(jnp.asarray(x))
        lift = geometry._coords_to_sym(a)
        g = _pade_expm(lift, self.order)
        return g @ x @ g.T
