from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import (
    RESULTS,
    AbstractReversibleSolver,
    AbstractStratonovichSolver,
    AbstractTerm,
)
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax_lowstorage import LowStorageRecurrence
from jaxtyping import PyTree

from georax._solver.commutator_free import AbstractLowStorageCommutatorFreeSolver
from georax._term import GeometricTerm
from diffrax_lowstorage.ees27 import _ees27_recurrence


_cf_ees27_recurrence = _ees27_recurrence

_SolverState = Y


class CFEES27(
    AbstractLowStorageCommutatorFreeSolver,
    AbstractReversibleSolver,
    AbstractStratonovichSolver,
):
    """Commutator-free EES(2,7;(5 - 3*sqrt(2))/14) solver.

    Supports ODEs and SDEs. For SDEs, this converges to the Stratonovich
    solution. O(1)-reversible and uses minimal memory and exponential count.

    ??? Reference

        ```bibtex
        @article{ShmelevThompsonSalvi2025,
          title = {Explicit and Effectively Symmetric Schemes for Neural SDEs on Lie Groups},
          author = {Shmelev, Daniil and Thompson, Luke and Salvi, Cristopher},
          year = {2025},
          doi = {10.48550/arXiv.2509.20599},
          url = {https://arxiv.org/abs/2509.20599}
        }
        ```
    """

    recurrence: ClassVar[LowStorageRecurrence] = _cf_ees27_recurrence

    @override
    def init(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        super().init(terms, t0, t1, y0, args)
        return y0

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 2

    def strong_order(self, terms):
        del terms
        return 0.5

    def antisymmetric_order(self, terms):
        del terms
        return 7

    @override
    def step(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, _SolverState, RESULTS]:
        y1, y_error, dense_info, _, result = super().step(
            terms, t0, t1, y0, args, None, made_jump
        )
        return y1, y_error, dense_info, y1, result

    @override
    def backward_step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y1: Y,
        args: Args,
        ts_state: PyTree[RealScalarLike],
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, _SolverState, RESULTS]:
        y0, _, dense_info, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, dense_info, solver_state, result
