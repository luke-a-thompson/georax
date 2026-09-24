from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, override

from diffrax import RESULTS, AbstractItoSolver, AbstractTerm

from georax._compat import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from georax._term import find_geometry

from ._interpolation import GeometricInterpolation, geometric_dense_info


class GeometricEuler(AbstractItoSolver):
    """Geometric Euler method.

    For SDE terms this is Euler--Maruyama: the full Diffrax term increment is
    evaluated at the left endpoint, then applied via the manifold chart.

    ??? Reference

        ```bibtex
        @book{KloedenPlaten1992,
          title = {Numerical Solution of Stochastic Differential Equations},
          author = {Kloeden, Peter E. and Platen, Eckhard},
          publisher = {Springer},
          year = {1992},
          doi = {10.1007/978-3-662-12616-5}
        }
        ```
    """

    term_structure: ClassVar[type[AbstractTerm]] = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., GeometricInterpolation]] = (
        GeometricInterpolation
    )

    def order(self, terms: AbstractTerm) -> int:
        return 1

    def strong_order(self, terms: AbstractTerm) -> float:
        return 0.5

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
    ) -> tuple[Y, None, DenseInfo, None, RESULTS]:
        del solver_state, made_jump

        geometry = find_geometry(terms)
        chart = geometry.select_chart(2)

        vf = terms.vf(t0, y0, args)
        control = terms.contr(t0, t1)
        increment = terms.prod(vf, control)
        y1 = geometry.apply_increment(y0, increment, chart)

        dense_info = geometric_dense_info(y0, y1, (increment,), geometry, chart)
        return y1, None, dense_info, None, RESULTS.successful
