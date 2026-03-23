from __future__ import annotations

from typing import ClassVar, override

import numpy as np

from georax._solver.base import AbstractCommutatorFreeSolver
from georax._term import GeometricTerm
from georax._solver.base import CommutatorFreeTableau

_cg2_tableau = CommutatorFreeTableau(
    c=(0.0, 0.5),
    stage_exps=(
        (),
        (np.array([0.5]),),
    ),
    final_exps=(np.array([0.0, 1.0]),),
)


class CG2(AbstractCommutatorFreeSolver):
    """Crouch-Grossman order-2 solver."""

    tableau: ClassVar[CommutatorFreeTableau] = _cg2_tableau

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 2
