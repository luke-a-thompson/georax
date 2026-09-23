from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, override

import equinox as eqx
from diffrax import (
    RESULTS,
    AbstractERK,
    AbstractTerm,
    AbstractWrappedSolver,
)

from georax._compat import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from georax._term import (
    GeometricTerm,
    Pullback,
    PulledDriftTerm,
    find_geometry,
    select_chart_for_solver,
)

from ._interpolation import GeometricInterpolation
from ._pullback import pullback_step


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

    # Equinox freezes this field; it implements Diffrax's AbstractVar.
    solver: AbstractERK = eqx.field(static=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    term_structure: ClassVar[type[GeometricTerm]] = GeometricTerm
    interpolation_cls: ClassVar[Callable[..., GeometricInterpolation]] = (
        GeometricInterpolation
    )

    def __init__(self, solver: AbstractERK) -> None:
        if not isinstance(solver, AbstractERK):
            raise TypeError("RKMK requires a base explicit Runge-Kutta solver.")
        # FSAL caches the previous step's last stage as f0 of the next, but
        # our y_anchor changes between RKMK steps so any cached value would be
        # stale. Disable FSAL on the wrapped solver to force a fresh first
        # stage every step.
        solver = eqx.tree_at(lambda s: s.disable_fsal, solver, True)
        object.__setattr__(self, "solver", solver)

    def order(self, terms: AbstractTerm) -> int | None:
        return self.solver.order(terms)

    @override
    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        del t0, t1, args
        find_geometry(terms).check_state_shape(y0)
        return None

    @override
    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    @override
    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, None, RESULTS]:
        del solver_state

        geometry = find_geometry(terms)
        chart = select_chart_for_solver(self, terms, geometry, pullback=True)
        pullback = Pullback(geometry, chart, y0)
        return pullback_step(
            self.solver,
            PulledDriftTerm(terms, pullback),
            pullback,
            t0,
            t1,
            args,
            made_jump,
        )
