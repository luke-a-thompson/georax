from __future__ import annotations

from typing import ClassVar, override

from diffrax import RESULTS, AbstractTerm
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solver.base import AbstractItoSolver

from georax._term import find_geometry


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

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar = LocalLinearInterpolation

    def order(self, terms) -> int:
        return 1

    def strong_order(self, terms) -> float:
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
        del t0, t1, y0, args
        geometry = find_geometry(terms)
        geometry.select_chart(2)
        return None

    @override
    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ):
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
        if geometry.chart is None:
            raise TypeError("GeometricEuler requires a geometry with a selected chart.")

        vf = terms.vf(t0, y0, args)
        control = terms.contr(t0, t1)
        increment = terms.prod(vf, control)
        y1 = geometry.apply_increment(y0, increment)

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful
