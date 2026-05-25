from __future__ import annotations

from collections.abc import Callable
from typing import Any, override, TypeVar

import equinox as eqx
from diffrax import AbstractSolver, AbstractTerm, MultiTerm
from diffrax._custom_types import VF, Args, RealScalarLike, Y
from diffrax._term import WrapTerm
from jaxtyping import Array

from georax._geometry import Manifold

_SolverState = TypeVar("_SolverState")


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
        f"Expected a GeometricTerm, or a MultiTerm containing a GeometricTerm; got {type(base).__name__}."
    )


class PulledDriftTerm(AbstractTerm[Array, RealScalarLike]):
    """Lie-algebra pullback of a manifold drift term anchored at ``y_anchor``."""

    drift_term: GeometricTerm
    y_anchor: Y

    @override
    def vf(self, t: RealScalarLike, omega: Array, args: Args) -> Array:
        geometry = self.drift_term.geometry
        chart = geometry.chart
        if chart is None:
            raise TypeError("Pullback term requires a geometry with a selected chart.")
        y = geometry.apply_increment(self.y_anchor, omega)
        raw = self.drift_term.vf(t, y, args)
        return chart.inverse_differential(self.y_anchor, omega, raw, geometry)

    @override
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        del kwargs
        return t1 - t0

    @override
    def prod(self, vf: Array, control: RealScalarLike) -> Array:
        return vf * control


class PulledDiffusionTerm(AbstractTerm[VF, object]):
    """Local Lie-algebra diffusion induced by an arbitrary Diffrax term.

    The underlying term still owns the vector-field/control product via
    ``prod``. We intentionally do not call its ``vf_prod``: Diffrax's
    ``ControlTerm.vf_prod`` validates against the manifold state shape, whereas
    georax diffusion terms return frame-coordinate data.
    """

    drift_term: GeometricTerm
    diffusion_term: AbstractTerm
    y_anchor: Y
    omega0: Array

    @override
    def vf(self, t: RealScalarLike, omega: Array, args: Args) -> VF:
        y = self.drift_term.geometry.apply_increment(self.y_anchor, omega)
        return self.diffusion_term.vf(t, y, args)

    @override
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs):
        return self.diffusion_term.contr(t0, t1, **kwargs)

    @override
    def prod(self, vf: VF, control: object) -> Array:
        geometry = self.drift_term.geometry
        chart = geometry.chart
        if chart is None:
            raise TypeError("Pullback term requires a geometry with a selected chart.")
        raw_increment = self.diffusion_term.prod(vf, control)
        return chart.inverse_differential(
            self.y_anchor, self.omega0, raw_increment, geometry
        )

    @override
    def vf_prod(
        self,
        t: RealScalarLike,
        omega: Array,
        args: Args,
        control: object,
    ) -> Array:
        geometry = self.drift_term.geometry
        chart = geometry.chart
        if chart is None:
            raise TypeError("Pullback term requires a geometry with a selected chart.")
        y = geometry.apply_increment(self.y_anchor, omega)
        raw_vf = self.diffusion_term.vf(t, y, args)
        raw_increment = self.diffusion_term.prod(raw_vf, control)
        return chart.inverse_differential(self.y_anchor, omega, raw_increment, geometry)


def select_chart_for_solver(
    solver: AbstractSolver[_SolverState], geometry: Manifold[Any]
) -> None:
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


def coordinate_shape_for_solver(
    solver: AbstractSolver[_SolverState], geometry: Manifold[Any]
) -> tuple[int, ...]:
    try:
        return geometry.coordinate_shape
    except NotImplementedError as exc:
        raise TypeError(
            f"{type(solver).__name__} requires a geometry with a fixed coordinate_shape; "
            f"{type(geometry).__name__} does not provide one."
        ) from exc
