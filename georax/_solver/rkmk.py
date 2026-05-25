from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
from diffrax import RESULTS
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solver.base import AbstractSolver, AbstractWrappedSolver
from diffrax._solver.runge_kutta import AbstractERK

from georax._term import (
    GeometricTerm,
    PulledDriftTerm,
    coordinate_shape_for_solver,
    find_geometry,
    select_chart_for_solver,
    unwrap_term,
)


class RKMK(AbstractWrappedSolver):
    """RKMK lift of a Diffrax explicit Runge-Kutta solver.

    The drift vector field is pulled back to the Lie algebra of the geometry's
    selected chart and integrated by the wrapped ERK starting from ``omega = 0``.
    The resulting algebra increment is retracted onto the manifold once.

    ??? Reference

        ```bibtex
        @article{MuntheKaas1998,
          title = {Runge-Kutta methods on Lie groups},
          author = {Munthe-Kaas, Hans},
          journal = {BIT Numerical Mathematics},
          volume = {38},
          number = {1},
          pages = {92--111},
          year = {1998},
          doi = {10.1007/BF02510919}
        }
        ```
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

        base_term = unwrap_term(terms)
        if not isinstance(base_term, GeometricTerm):
            raise TypeError("RKMK requires a geometric drift term.")
        geometry = base_term.geometry

        algebra_term = PulledDriftTerm(base_term, y0)
        omega0 = jnp.zeros(
            coordinate_shape_for_solver(self, geometry), dtype=jnp.result_type(y0)
        )

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
