from __future__ import annotations

from typing import ClassVar, override

import numpy as np
from diffrax_lowstorage import LowStorageRecurrence

from georax import GeometricTerm
from georax._solver.base import AbstractLowStorageCommutatorFreeSolver

_cf_ees25_recurrence = LowStorageRecurrence(
    A=np.array([-0.5, -2.0]),
    B=np.array([0.5, 1.0, 0.25]),
    C=np.array([0.0, 0.5, 1.0]),
)


class CFEES25(AbstractLowStorageCommutatorFreeSolver):
    """Commutator-free EES(2,5;1/4) solver with chained exponentials."""

    recurrence: ClassVar[LowStorageRecurrence] = _cf_ees25_recurrence

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 2
