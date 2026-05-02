from __future__ import annotations

from typing import ClassVar, override

from diffrax import RESULTS, AbstractTerm, MultiTerm
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solver.base import AbstractItoSolver
from diffrax._term import WrapTerm

from georax._term import GeometricTerm


class GeometricEuler(AbstractItoSolver):
    """Geometric Euler method.

    For SDE terms this is Euler--Maruyama: the full Diffrax term increment is
    evaluated at the left endpoint, then applied via the manifold chart.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar = LocalLinearInterpolation

    def order(self, terms) -> int:
        return 1

    def strong_order(self, terms) -> float:
        return 0.5

    @staticmethod
    def _unwrap_term(term: AbstractTerm) -> AbstractTerm:
        while isinstance(term, WrapTerm):
            term = term.term
        return term

    @classmethod
    def _geometric_term(cls, terms: AbstractTerm) -> GeometricTerm:
        terms = cls._unwrap_term(terms)
        if isinstance(terms, GeometricTerm):
            return terms
        if isinstance(terms, MultiTerm) and len(terms.terms) > 0:
            first_term = cls._unwrap_term(terms.terms[0])
            if isinstance(first_term, GeometricTerm):
                return first_term
        raise TypeError(
            "GeometricEuler expects a GeometricTerm, or a MultiTerm whose first "
            "term is a GeometricTerm."
        )

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
        geometric_term = self._geometric_term(terms)
        geometric_term.geometry.select_chart(2)
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

        geometric_term = self._geometric_term(terms)
        if geometric_term.geometry.chart is None:
            raise TypeError("GeometricEuler requires a geometry with a selected chart.")

        vf = terms.vf(t0, y0, args)
        control = terms.contr(t0, t1)
        increment = terms.prod(vf, control)
        y1 = geometric_term.apply_increment(y0, increment)

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful
