from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array

from .base import LocalChart

if TYPE_CHECKING:
    from .spd import SPD
    from .special_orthogonal import SO


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


def _cayley(a: Array) -> Array:
    ident = jnp.eye(a.shape[0], dtype=a.dtype)
    return jnp.linalg.solve(ident - 0.5 * a, ident + 0.5 * a)


# ---------------------------------------------------------------------------
# SO(n) charts
# ---------------------------------------------------------------------------


class SOChart(LocalChart["SO"]):
    """SO(n) chart using Cayley at order 2 and Taylor+QR at higher orders."""

    order: int

    def __init__(self, order: int):
        object.__setattr__(self, "order", int(order))

    def apply(self, x: Array, a: Array, geometry: SO) -> Array:
        omega = geometry._coords_to_alg(a)
        if self.order <= 2:
            return x @ _cayley(omega)

        degree = max(2, self.order)
        if degree % 2:
            degree += 1
        g = _taylor_expm(omega, degree)
        q, r = jnp.linalg.qr(g)
        q = q * jnp.sign(jnp.diag(r))
        return x @ q

    def inverse_differential(self, x: Array, a: Array, b: Array, geometry: SO) -> Array:
        if self.order > 2:
            raise NotImplementedError(
                "SOChart only implements inverse_differential for the order-2 Cayley chart."
            )
        del x
        omega = geometry._coords_to_alg(a)
        eta = geometry._coords_to_alg(b)
        ident = jnp.eye(geometry.n, dtype=eta.dtype)
        corrected = (ident + 0.5 * omega) @ eta @ (ident - 0.5 * omega)
        return geometry._alg_to_coords(corrected)


# ---------------------------------------------------------------------------
# SPD(n) charts
# ---------------------------------------------------------------------------


class SPDChart(LocalChart["SPD"]):
    order: int

    def __init__(self, order: int):
        object.__setattr__(self, "order", int(order))

    def apply(self, x: Array, a: Array, geometry: SPD) -> Array:
        x = _sym(jnp.asarray(x))
        lift = geometry._coords_to_sym(a)
        degree = max(2, self.order)
        if degree % 2:
            degree += 1
        g = _taylor_expm(lift, degree)
        return g @ x @ g.T
