from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import AbstractTerm, ControlTerm, MultiTerm, ODETerm
from jaxtyping import Array

from georax import CFEES25, CFEES27, CG2, CG4, SO, GeometricTerm
from georax._compat import Args, RealScalarLike

# ── Solvers ────────────────────────────────────────────────────────────────────

SOLVERS = [
    ("cg2", CG2),
    ("cg4", CG4),
    ("cfees25", CFEES25),
    ("cfees27", CFEES27),
]

# ── SO(3) accuracy terms (used by tests and docs examples) ────────────────────

_SO3_GEOMETRY = SO(3)
_SO3_NOISE_SCALE = jnp.array([0.12, 0.08, 0.05], dtype=jnp.float32)


def so3_body_omega(t: RealScalarLike) -> Array:
    t_array = jnp.asarray(t)
    return jnp.array(
        [
            0.8 + 0.45 * jnp.sin(0.7 * t_array),
            0.55 * jnp.cos(1.3 * t_array + 0.2),
            0.35 + 0.6 * jnp.sin(0.9 * t_array - 0.4),
        ]
    )


def so3_body_frame_coeffs(t: RealScalarLike) -> Array:
    omega = so3_body_omega(t)
    return jnp.array([-omega[2], omega[1], -omega[0]], dtype=omega.dtype)


def make_solver_accuracy_term() -> GeometricTerm:
    def coeffs(t: RealScalarLike, R: Array, args: Args) -> Array:
        del R, args
        return so3_body_frame_coeffs(t)

    return GeometricTerm(coeffs, geometry=_SO3_GEOMETRY)


def make_solver_accuracy_ambient_term() -> ODETerm:
    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        return R @ _SO3_GEOMETRY._coords_to_alg(so3_body_frame_coeffs(t))

    return diffrax.ODETerm(vf)


def make_solver_accuracy_sde_term(
    *,
    bm_tol: float = 1e-3,
    key: Array | None = None,
) -> AbstractTerm:
    if key is None:
        key = jax.random.key(0)

    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        tol=bm_tol,
        shape=_SO3_GEOMETRY.coordinate_shape,
        key=key,
    )

    def coeffs(t: RealScalarLike, R: Array, args: Args) -> Array:
        del R, args
        return so3_body_frame_coeffs(t)

    def diffusion(t: RealScalarLike, R: Array, args: Args):
        del t, R, args
        return lx.DiagonalLinearOperator(_SO3_NOISE_SCALE)

    return MultiTerm(
        GeometricTerm(coeffs, geometry=_SO3_GEOMETRY),
        ControlTerm(diffusion, brownian),
    )
