from __future__ import annotations

from typing import ClassVar, override

import numpy as np

from georax._solver.commutator_free import AbstractCommutatorFreeSolver
from georax._term import GeometricTerm
from georax._solver.commutator_free import CommutatorFreeTableau

_cg2_tableau = CommutatorFreeTableau(
    c=(0.0, 0.5),
    stage_exps=(
        (),
        (np.array([0.5]),),
    ),
    final_exps=(np.array([0.0, 1.0]),),
)


class CG2(AbstractCommutatorFreeSolver):
    """Crouch-Grossman order-2 solver.

    ??? Reference

        ```bibtex
        @article{CrouchGrossman1993,
          title = {Numerical integration of ordinary differential equations on manifolds},
          author = {Crouch, P. E. and Grossman, R.},
          journal = {Journal of Nonlinear Science},
          volume = {3},
          number = {1},
          pages = {1--33},
          year = {1993},
          doi = {10.1007/BF02429858}
        }
        ```
    """

    tableau: ClassVar[CommutatorFreeTableau] = _cg2_tableau

    @override
    def order(self, terms: GeometricTerm) -> int:
        del terms
        return 2
