from __future__ import annotations

import diffrax
import jax.numpy as jnp
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array

from georax import CFEES25, CG2, CG4, SO, GeometricTerm, LocalFlow, Manifold

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

    def select_flow_method(self, required_order: RealScalarLike) -> LocalFlow:
        del required_order
        flow: LocalFlow = _EuclideanFlow()
        object.__setattr__(self, "flow", flow)
        return flow


class _EuclideanFlow(LocalFlow):
    order: int | str = "exact"
    inverse_order: int | str = "exact"

    def forward(self, x: Array, a: Array, geometry: EuclideanOps) -> Array:
        return geometry.retraction(x, geometry.from_frame(x, a))


def make_solver_accuracy_term() -> GeometricTerm:
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

    return GeometricTerm(inner=diffrax.ODETerm(vf), geometry=SO(3))
