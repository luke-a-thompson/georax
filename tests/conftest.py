from __future__ import annotations

import diffrax
import jax.numpy as jnp
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array

from georax import CFEES25, CG2, CG4, GeometricTerm, LieGroup, Manifold

SOLVERS = [
    ("cg2", CG2),
    ("cg4", CG4),
    ("cfees25", CFEES25),
]

BENCH_SOLVERS = SOLVERS


class EuclideanOps(Manifold):
    def frame(self, x: Array) -> Array:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    def retraction(self, x: Array, v: Array) -> Array:
        return x + v


def make_solver_accuracy_term() -> GeometricTerm:
    class _SO3ExactExp(LieGroup):
        @property
        def lie_algebra_dimension(self) -> int:
            return 3

        def frame(self, x: Array) -> Array:
            basis = jnp.stack(
                [
                    jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
                    jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                    jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                ],
                axis=-1,
            )
            return jnp.einsum("ab,bck->ack", x, basis.astype(x.dtype))

        def to_frame(self, x: Array, v: Array) -> Array:
            omega = x.T @ v
            return jnp.array([omega[2, 1], omega[0, 2], omega[1, 0]], dtype=v.dtype)

        def from_frame(self, x: Array, a: Array) -> Array:
            a = jnp.asarray(a, dtype=x.dtype)
            omega = jnp.array(
                [
                    [0.0, -a[2], a[1]],
                    [a[2], 0.0, -a[0]],
                    [-a[1], a[0], 0.0],
                ],
                dtype=x.dtype,
            )
            return x @ omega

        def retraction(self, x: Array, v: Array) -> Array:
            omega = x.T @ v
            theta_sq = jnp.sum(omega * omega) / 2.0
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
            ident = jnp.eye(3, dtype=x.dtype)
            exp_omega = (
                ident
                + sin_over_theta * omega
                + one_minus_cos_over_theta_sq * (omega @ omega)
            )
            return x @ exp_omega

        def chart_differential_inv(self, a: Array, b: Array) -> Array:
            del a
            return b

    def omega_body(t: RealScalarLike) -> Array:
        t_array = jnp.asarray(t)
        return jnp.array(
            [
                0.8 + 0.45 * jnp.sin(0.7 * t_array),
                0.55 * jnp.cos(1.3 * t_array + 0.2),
                0.35 + 0.6 * jnp.sin(0.9 * t_array - 0.4),
            ]
        )

    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        omega = omega_body(t)
        skew = jnp.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ]
        )
        return R @ skew

    return GeometricTerm(inner=diffrax.ODETerm(vf), geometry=_SO3ExactExp())
