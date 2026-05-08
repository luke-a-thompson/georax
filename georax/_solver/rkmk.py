from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
from diffrax import RESULTS, AbstractTerm
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solver.base import AbstractSolver, AbstractWrappedSolver
from diffrax._solver.runge_kutta import AbstractERK
from jaxtyping import Array

from georax._term import GeometricTerm, find_geometry, select_chart_for_solver


class _PulledTerm(AbstractTerm[Array, RealScalarLike]):
    """Lie-algebra pullback of a manifold drift term anchored at ``y_anchor``."""

    drift_term: GeometricTerm
    y_anchor: Y

    @override
    def vf(self, t: RealScalarLike, omega: Array, args: Args) -> Array:
        geometry = self.drift_term.geometry
        chart = geometry.chart
        if chart is None:
            raise TypeError("RKMK requires a geometry with a selected chart.")
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


class RKMK(AbstractWrappedSolver):
    """RKMK lift of a Diffrax explicit Runge-Kutta solver.

    The drift vector field is pulled back to the Lie algebra of the geometry's
    selected chart and integrated by the wrapped ERK starting from ``omega = 0``.
    The resulting algebra increment is retracted onto the manifold once.
    """

    solver: AbstractSolver = eqx.field(static=True)

    def __init__(self, solver: AbstractSolver):
        if not isinstance(solver, AbstractERK):
            raise TypeError("RKMK requires a base explicit Runge-Kutta solver.")
        # FSAL caches the previous step's last stage as f0 of the next, but
        # our y_anchor changes between RKMK steps so any cached value would be
        # stale. Disable FSAL on the wrapped solver to force a fresh first
        # stage every step.
        solver = eqx.tree_at(lambda s: s.disable_fsal, solver, True)
        object.__setattr__(self, "solver", solver)

    @property
    def term_structure(self):  # pyright: ignore
        return GeometricTerm

    @property
    def interpolation_cls(self):  # pyright: ignore
        # Stages live in the algebra; do not inherit Euclidean Hermite
        # interpolation from the wrapped solver.
        return LocalLinearInterpolation

    def order(self, terms) -> int | None:
        return self.solver.order(terms)

    @override
    def init(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        del t0, t1, y0, args
        select_chart_for_solver(self, find_geometry(terms))
        return None

    @override
    def func(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    @override
    def step(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, None, RESULTS]:
        del solver_state

        geometry = find_geometry(terms)
        if geometry.chart is None:
            raise TypeError("RKMK requires a GeometricTerm geometry with a chart.")

        algebra_term = _PulledTerm(terms, y0)
        omega0 = jnp.zeros_like(terms.vf(t0, y0, args))

        omega1, omega_error, _, _, result = self.solver.step(
            algebra_term,
            t0,
            t1,
            omega0,
            args,
            None,
            made_jump,
        )
        y1 = geometry.apply_increment(y0, omega1)

        y_error = None
        if omega_error is not None:
            y_error = geometry.apply_increment(y0, omega_error) - y0

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, result
