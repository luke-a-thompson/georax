from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax import AbstractReversibleSolver
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._solution import RESULTS
from diffrax._term import AbstractTerm
from diffrax_lowstorage import LowStorageRecurrence
from jaxtyping import PyTree

from georax._solver.commutator_free import AbstractLowStorageCommutatorFreeSolver
from georax._term import GeometricTerm

_cf_ees27_recurrence = LowStorageRecurrence(
    A=np.array([1.0 - np.sqrt(2.0), -1.0, -(1.0 + np.sqrt(2.0))]),
    B=np.array(
        [
            0.5 * (2.0 - np.sqrt(2.0)),
            0.5 * np.sqrt(2.0),
            0.5 * np.sqrt(2.0),
            0.25 * (2.0 - np.sqrt(2.0)),
        ]
    ),
    C=np.array(
        [
            0.0,
            0.5 * (2.0 - np.sqrt(2.0)),
            0.5 * np.sqrt(2.0),
            1.0,
        ]
    ),
)

_SolverState = Y


class CFEES27(AbstractLowStorageCommutatorFreeSolver, AbstractReversibleSolver):
    """Commutator-free EES(2,7;1/4) solver with chained exponentials."""

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
        del terms, t0, t1, args
        return y0

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 2

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
