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

_cf_ees25_recurrence = LowStorageRecurrence(
    A=np.array([-0.5, -2.0]),
    B=np.array([0.5, 1.0, 0.25]),
    C=np.array([0.0, 0.5, 1.0]),
)

_SolverState = Y


class CFEES25(AbstractLowStorageCommutatorFreeSolver, AbstractReversibleSolver):
    """Commutator-free EES(2,5;1/4) solver with chained exponentials."""

    recurrence: ClassVar[LowStorageRecurrence] = _cf_ees25_recurrence

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 5

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
