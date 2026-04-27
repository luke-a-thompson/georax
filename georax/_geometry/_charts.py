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


def _quadratic_expm(a: Array) -> Array:
    # Degree-2 Taylor: I + A + A^2/2. 1 matmul.
    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    half = jnp.asarray(0.5, dtype=a.dtype)
    return ident + a + half * (a @ a)


def _bbc_expm_4(a: Array) -> Array:
    # Sastre/BBC scheme: 2 matmuls, exact match to Taylor degree 4.
    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    half = jnp.asarray(0.5, dtype=a.dtype)
    c3 = jnp.asarray(1.0 / 6.0, dtype=a.dtype)
    c4 = jnp.asarray(1.0 / 24.0, dtype=a.dtype)
    a2 = a @ a
    m = a2 @ (half * ident + c3 * a + c4 * a2)
    return ident + a + m


def _bbc_expm_8(a: Array) -> Array:
    # Sastre/BBC-style scheme: 3 matmuls, exact match to Taylor degree 8.
    # Free parameter chosen as x1=1, x2=1/4 (giving x3 = (-1+sqrt(177))/2).
    sqrt177 = math.sqrt(177.0)
    x1_v = 1.0
    x2_v = 0.25
    x3_v = (-1.0 + sqrt177) / 2.0
    x5_v = 11.0 / 630.0
    x7_v = 1.0 / 2520.0
    x4_v = 1.0 / 6.0 - (11.0 / 630.0) * x3_v
    x6_v = (10.0 - x3_v) / 2520.0
    y2_v = 0.5 - x3_v * x4_v

    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    x1 = jnp.asarray(x1_v, dtype=a.dtype)
    x2 = jnp.asarray(x2_v, dtype=a.dtype)
    x3 = jnp.asarray(x3_v, dtype=a.dtype)
    x4 = jnp.asarray(x4_v, dtype=a.dtype)
    x5 = jnp.asarray(x5_v, dtype=a.dtype)
    x6 = jnp.asarray(x6_v, dtype=a.dtype)
    x7 = jnp.asarray(x7_v, dtype=a.dtype)
    y2 = jnp.asarray(y2_v, dtype=a.dtype)

    a2 = a @ a
    a4 = a2 @ (x1 * a + x2 * a2)
    a8 = (x3 * a2 + a4) @ (x4 * ident + x5 * a + x6 * a2 + x7 * a4)
    return ident + a + y2 * a2 + a8


def _ps_expm_12(a: Array) -> Array:
    # Paterson-Stockmeyer with s=4: 6 matmuls, exact match to Taylor degree 12.
    # BBC's 4-matmul scheme for degree 12 requires solving a nonlinear system;
    # this PS variant gives a 2x speedup over Horner without that complexity.
    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    c = [jnp.asarray(1.0 / math.factorial(k), dtype=a.dtype) for k in range(13)]
    a2 = a @ a
    a3 = a2 @ a
    a4 = a2 @ a2
    p = c[12] * ident
    p = (c[8] * ident + c[9] * a + c[10] * a2 + c[11] * a3) + p @ a4
    p = (c[4] * ident + c[5] * a + c[6] * a2 + c[7] * a3) + p @ a4
    p = (c[0] * ident + c[1] * a + c[2] * a2 + c[3] * a3) + p @ a4
    return p


def _horner_taylor_expm(a: Array, degree: int) -> Array:
    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    result = jnp.asarray(1 / math.factorial(degree), dtype=a.dtype) * ident
    for k in range(degree - 1, -1, -1):
        result = a @ result
        result = result + jnp.asarray(1 / math.factorial(k), dtype=a.dtype) * ident
    return result


def _taylor_expm(a: Array, degree: int) -> Array:
    # Static dispatch: `degree` is fixed at chart construction, so Python
    # branching here selects one scheme at trace time and JAX sees a fully
    # unrolled graph for that scheme alone.
    degree = int(degree)
    if degree <= 2:
        return _quadratic_expm(a)
    if degree <= 4:
        return _bbc_expm_4(a)
    if degree <= 8:
        return _bbc_expm_8(a)
    if degree <= 12:
        return _ps_expm_12(a)
    return _horner_taylor_expm(a, degree)


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


class CongruenceTaylorChart(LocalChart):
    order: chart_order
    inverse_order: chart_order
    degree: int

    def __init__(self, order: int):
        order = int(order)
        degree = max(2, order)
        if degree % 2:
            degree += 1
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "inverse_order", order)
        object.__setattr__(self, "degree", degree)

    def apply(self, x: Array, a: Array, geometry: SPD) -> Array:
        x = _sym(jnp.asarray(x))
        lift = geometry._coords_to_sym(a)
        g = _taylor_expm(lift, self.degree)
        return g @ x @ g.T
