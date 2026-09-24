from __future__ import annotations

from typing import ClassVar

from diffrax import AbstractTerm
from diffrax_lowstorage import EES25, LowStorageRecurrence

from georax._solver.commutator_free import _AbstractCFEES


class CFEES25(_AbstractCFEES):
    """Commutator-free EES(2,5;1/10) solver.

    Supports ODEs and SDEs. For SDEs, this converges to the Stratonovich
    solution. O(1)-reversible and uses minimal memory and exponential count.

    Includes a first-order embedded ODE companion using the normalised final
    increment: one additional chart action and no extra generator evaluations.
    Returns an ambient endpoint difference for Diffrax's adaptive controllers.

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

    recurrence: ClassVar[LowStorageRecurrence] = EES25.recurrence

    def antisymmetric_order(self, terms: AbstractTerm) -> int:
        del terms
        return 5
