from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import equinox as eqx
from diffrax import AbstractTerm, MultiTerm
from diffrax._custom_types import Args, RealScalarLike, Y
from diffrax._term import WrapTerm
from jaxtyping import Array

from georax._geometry import Manifold


class GeometricTerm(AbstractTerm[Array, RealScalarLike]):
    """Intrinsic manifold term whose vector field is given in frame coordinates."""

    geometry: Manifold[Any]
    coeffs_fn: Callable[[RealScalarLike, Array, Args], Array] = eqx.field(static=True)

    def __init__(
        self,
        coeffs: Callable[[RealScalarLike, Array, Args], Array],
        geometry: Manifold[Any],
    ):
        object.__setattr__(self, "coeffs_fn", coeffs)
        object.__setattr__(self, "geometry", geometry)

    @override
    def vf(self, t: RealScalarLike, y: Y, args: Args) -> Array:
        return self.coeffs_fn(t, y, args)

    @override
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        del kwargs
        return t1 - t0

    @override
    def prod(self, vf: Array, control: RealScalarLike) -> Y:
        return vf * control


def unwrap_term(term: AbstractTerm) -> AbstractTerm:
    """Strip diffrax ``WrapTerm`` wrappers."""
    while isinstance(term, WrapTerm):
        term = term.term
    return term


def find_geometry(terms: AbstractTerm) -> Manifold[Any]:
    """Locate the geometry inside a possibly wrapped geometric term."""
    base = unwrap_term(terms)
    if isinstance(base, GeometricTerm):
        return base.geometry
    if isinstance(base, MultiTerm):
        for child in base.terms:
            child = unwrap_term(child)
            if isinstance(child, GeometricTerm):
                return child.geometry
    raise TypeError(
        "Expected a GeometricTerm, or a MultiTerm containing a GeometricTerm; "
        f"got {type(base).__name__}."
    )


def select_chart_for_solver(solver, geometry: Manifold[Any]) -> None:
    """Select a chart based on the highest order the solver may need."""
    orders = [
        getattr(solver, name, lambda _: None)(geometry)
        for name in ("order", "error_order", "antisymmetric_order")
    ]
    orders = [int(o) for o in orders if o is not None]
    if not orders:
        raise ValueError(
            f"Solver {type(solver).__name__} provides no order for chart selection."
        )
    geometry.select_chart(max(orders))
