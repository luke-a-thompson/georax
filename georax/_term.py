from __future__ import annotations

from collections.abc import Callable
from typing import override

import equinox as eqx
from diffrax import AbstractTerm
from diffrax._custom_types import Args, RealScalarLike, Y
from jaxtyping import Array

from georax._geometry import Manifold


class GeometricTerm(AbstractTerm[Array, RealScalarLike]):
    """Intrinsic manifold term whose vector field is given in frame coordinates."""

    geometry: Manifold
    coeffs_fn: Callable[[RealScalarLike, Array, Args], Array] = eqx.field(static=True)

    def __init__(
        self, coeffs: Callable[[RealScalarLike, Array, Args], Array], geometry: Manifold
    ):
        object.__setattr__(self, "coeffs_fn", coeffs)
        object.__setattr__(self, "geometry", geometry)

    @override
    def vf(self, t: RealScalarLike, y: Y, args: Args) -> Array:
        return self.coeffs(t, y, args)

    @override
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        del kwargs
        return t1 - t0

    @override
    def prod(self, vf: Array, control: RealScalarLike) -> Y:
        return vf * control

    @override
    def vf_prod(
        self, t: RealScalarLike, y: Y, args: Args, control: RealScalarLike
    ) -> Array:
        return self.prod(self.vf(t, y, args), control)

    def coeffs(self, t: RealScalarLike, x: Array, args: Args) -> Array:
        """Return intrinsic frame coefficients at ``(t, x)``."""
        return self.coeffs_fn(t, x, args)

    def apply_increment(self, x: Array, a: Array) -> Array:
        """Apply one intrinsic frame-coordinate increment via the geometry."""
        return self.geometry.apply_increment(x, a)


def select_chart_for_solver(solver, geometric_term: GeometricTerm) -> None:
    """Select a chart based on the highest order the solver may need."""
    orders = [
        getattr(solver, name, lambda _: None)(geometric_term)
        for name in ("order", "error_order", "antisymmetric_order")
    ]
    orders = [int(o) for o in orders if o is not None]
    if not orders:
        raise ValueError(
            f"Solver {type(solver).__name__} provides no order for chart selection."
        )
    geometric_term.geometry.select_chart(max(orders))
