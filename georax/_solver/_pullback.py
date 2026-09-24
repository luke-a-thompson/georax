from __future__ import annotations

from typing import Any

from diffrax import RESULTS, AbstractSolver, AbstractTerm
from jaxtyping import Array

from georax._compat import Args, BoolScalarLike, DenseInfo, RealScalarLike
from georax._term import Pullback

from ._interpolation import geometric_dense_info


def pullback_step(
    solver: AbstractSolver,
    terms: AbstractTerm,
    pullback: Pullback[Any],
    t0: RealScalarLike,
    t1: RealScalarLike,
    args: Args,
    made_jump: BoolScalarLike,
) -> tuple[Array, Array | None, DenseInfo, None, RESULTS]:
    """Integrate one local equation and reconstruct its solution and error."""
    omega, omega_error, _, _, result = solver.step(
        terms, t0, t1, pullback.zero(), args, None, made_jump
    )
    y1 = pullback.apply(omega)
    y_error = None
    if omega_error is not None:
        y_error = pullback.apply(omega + omega_error) - y1
    dense_info = geometric_dense_info(
        pullback.y_anchor, y1, (omega,), pullback.geometry, pullback.chart
    )
    return y1, y_error, dense_info, None, result
