from __future__ import annotations

from typing import override

import equinox as eqx
from diffrax import (
    RESULTS,
    AbstractERK,
    AbstractSolver,
    AbstractWrappedSolver,
    LocalLinearInterpolation,
)
from georax._compat import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from georax._term import (
    GeometricTerm,
    PulledDriftTerm,
    find_geometry,
    select_chart_for_solver,
    unwrap_term,
)


class RKMK(AbstractWrappedSolver):
    """RKMK lift of a Diffrax explicit Runge-Kutta solver.

    The drift vector field is pulled back to the Lie algebra of the geometry's
    selected chart and integrated by the wrapped ERK starting from ``omega = 0``.
    The resulting algebra increment is retracted onto the manifold once.

    On ``SO(n)``, the Cayley transform is used as an exact local coordinate map
    at every wrapped solver order. Its closed-form inverse differential preserves
    the wrapped Runge--Kutta order without treating Cayley as a high-order
    approximation to the exponential; see Iserles and Zanna (2000).

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

        @article{IserlesZanna2000,
          title = {On the Dimension of Certain Graded Lie Algebras Arising in Geometric Integration of Differential Equations},
          author = {Iserles, Arieh and Zanna, Antonella},
          journal = {LMS Journal of Computation and Mathematics},
          volume = {3},
          pages = {44--75},
          year = {2000},
          doi = {10.1112/S1461157000000206}
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
        select_chart_for_solver(self, terms, find_geometry(terms), pullback=True)
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
        omega0 = geometry.zero_coordinates(y0)

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
            y_hat = geometry.apply_increment(y0, omega1 + omega_error)
            y_error = y_hat - y1

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, result
