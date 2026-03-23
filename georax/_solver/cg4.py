from typing import ClassVar, override

import numpy as np

from georax._term import GeometricTerm
from georax._solver.commutator_free import (
    AbstractCommutatorFreeSolver,
    CommutatorFreeTableau,
)

_cg4_tableau = CommutatorFreeTableau(
    c=(
        0.0,
        0.8177227988124852,
        0.3859740639032449,
        0.3242290522866937,
        0.8768903263420429,
    ),
    stage_exps=(
        (),
        (np.array([0.8177227988124852]),),
        (
            np.array([0.3199876375476427, 0.0]),
            np.array([0.0, 0.0659864263556022]),
        ),
        (
            np.array([0.9214417194464946, 0.0, 0.0]),
            np.array([0.0, 0.4997857776773573, 0.0]),
            np.array([0.0, 0.0, -1.0969984448371582]),
        ),
        (
            np.array([0.3552358559023322, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.2390958372307326, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.3918565724203246, 0.0]),
            np.array([0.0, 0.0, 0.0, -1.1092979392113565]),
        ),
    ),
    final_exps=(
        np.array([0.1370831520630755, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.0183698531564020, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.7397813985370780, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, -0.1907142565505889, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.3322195591068374]),
    ),
)


class CG4(AbstractCommutatorFreeSolver):
    """Crouch-Grossman order-4 solver."""

    tableau: ClassVar[CommutatorFreeTableau] = _cg4_tableau

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 4
