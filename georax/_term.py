from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, override

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractSolver, AbstractTerm, MultiTerm
from jaxtyping import Array

from georax._compat import VF, Args, Control, RealScalarLike, WrapTerm
from georax._geometry import LocalChart, Manifold
from georax._geometry.base import Geometry


class GeometricTerm(AbstractTerm[Array, RealScalarLike]):
    """Intrinsic manifold term whose vector field is given in frame coordinates."""

    geometry: Manifold[Any]
    coeffs_fn: Callable[[RealScalarLike, Array, Args], Array]

    def __init__(
        self,
        coeffs: Callable[[RealScalarLike, Array, Args], Array],
        geometry: Manifold[Any],
    ) -> None:
        object.__setattr__(self, "coeffs_fn", coeffs)
        object.__setattr__(self, "geometry", geometry)

    @override
    def vf(self, t: RealScalarLike, y: Array, args: Args) -> Array:
        return self.coeffs_fn(t, y, args)

    @override
    def contr(
        self, t0: RealScalarLike, t1: RealScalarLike, **kwargs: Any
    ) -> RealScalarLike:
        del kwargs
        return t1 - t0

    @override
    def prod(self, vf: Array, control: RealScalarLike) -> Array:
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
        f"Expected a GeometricTerm, or a MultiTerm containing a GeometricTerm; got {type(base).__name__}."
    )


class Pullback(eqx.Module, Generic[Geometry]):
    """Coordinates anchored at one manifold state for a single solver step."""

    geometry: Geometry
    chart: LocalChart[Geometry]
    y_anchor: Array

    def apply(self, omega: Array) -> Array:
        return self.geometry.apply_increment(self.y_anchor, omega, self.chart)

    def inverse_differential(self, omega: Array, tangent: Array) -> Array:
        return self.chart.inverse_differential(
            self.y_anchor, omega, tangent, self.geometry
        )

    def zero(self) -> Array:
        return self.geometry.zero_coordinates(self.y_anchor)


class PulledDriftTerm(AbstractTerm[Array, RealScalarLike]):
    """Drift in local coordinates and in Diffrax's normalized time direction."""

    term: AbstractTerm
    pullback: Pullback[Any]

    def vf(self, t: RealScalarLike, y: Array, args: Args) -> Array:
        raw = self.term.vf(t, self.pullback.apply(y), args)
        # SRK methods multiply drift by their own positive step duration. Put
        # the direction in vf so both ERK and SRK evaluate the same equation.
        term = self.term
        while isinstance(term, WrapTerm):
            raw = raw * term.direction
            term = term.term
        return self.pullback.inverse_differential(y, raw)

    def contr(
        self, t0: RealScalarLike, t1: RealScalarLike, **kwargs: Any
    ) -> RealScalarLike:
        return t1 - t0

    def prod(self, vf: Array, control: RealScalarLike) -> Array:
        return vf * control


class PulledDiffusionTerm(AbstractTerm[VF, Control]):
    """Diffusion pulled back at the current stage, with a linear vf/prod pair.

    vf materializes the control-to-coordinate map for solvers that combine
    vector fields. vf_prod contracts first to retain efficient operator-valued
    diffusion for general SRKs. Both use the same stage-dependent pullback.
    """

    term: AbstractTerm
    pullback: Pullback[Any]

    def _prod_at(self, omega: Array, raw_vf: VF, control: Control) -> Array:
        return self.pullback.inverse_differential(
            omega, self.term.prod(raw_vf, control)
        )

    def vf(self, t: RealScalarLike, y: Array, args: Args) -> VF:
        raw_vf = self.term.vf(t, self.pullback.apply(y), args)
        control_shape = eqx.filter_eval_shape(self.term.contr, t, t)
        zero = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), control_shape)
        return jax.jacfwd(lambda control: self._prod_at(y, raw_vf, control))(zero)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs: Any) -> Control:
        return self.term.contr(t0, t1, **kwargs)

    def prod(self, vf: VF, control: Control) -> Array:
        products = jax.tree.map(
            lambda matrix, value: jnp.tensordot(matrix, value, axes=jnp.ndim(value)),
            vf,
            control,
        )
        return sum(jax.tree.leaves(products), self.pullback.zero())

    def vf_prod(
        self, t: RealScalarLike, y: Array, args: Args, control: Control
    ) -> Array:
        raw_vf = self.term.vf(t, self.pullback.apply(y), args)
        return self._prod_at(y, raw_vf, control)


def select_chart_for_solver(
    solver: AbstractSolver,
    terms: AbstractTerm,
    geometry: Geometry,
    *,
    pullback: bool = False,
) -> LocalChart[Geometry]:
    """Select a chart based on the highest order the solver may need."""
    orders = [
        getattr(solver, name, lambda _: None)(terms)
        for name in ("order", "error_order", "antisymmetric_order")
    ]
    orders = [int(o) for o in orders if o is not None]
    if not orders:
        raise ValueError(
            f"Solver {type(solver).__name__} provides no order for chart selection."
        )
    selector = geometry.select_pullback_chart if pullback else geometry.select_chart
    return selector(max(orders))
