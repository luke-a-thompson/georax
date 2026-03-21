from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from georax import CFEES25, CG2, GeometricOps

SOLVERS = [
    ("cg2", CG2),
    ("cfees25", CFEES25),
]

BENCH_SOLVERS = SOLVERS


class EuclideanOps(GeometricOps):
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
