from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from diffrax import AbstractTerm
from diffrax_lowstorage import CosineEES, LowStorageRecurrence

from georax._solver.commutator_free import _AbstractCFEES


class CFCosineEES(_AbstractCFEES):
    """Second-order commutator-free cosine EES solver in 2N storage.

    `reversible_order` must be an odd integer >= 3; the method uses
    `(reversible_order + 1) // 2` stages. Supports ODEs and Stratonovich SDEs,
    with strong order 0.5, and the reversible adjoint.

    No embedded error estimate is provided; use fixed step sizes.
    """

    reversible_order: int = eqx.field(static=True)
    # The cosine recurrence's final accumulator has zero normalization, so
    # the embedded companion used by CFEES25 and CFEES27 does not apply.
    embedded_final_increment: ClassVar[bool] = False

    def __check_init__(self) -> None:
        CosineEES(self.reversible_order)

    @property
    def recurrence(self) -> LowStorageRecurrence:
        return CosineEES(self.reversible_order).recurrence

    def antisymmetric_order(self, terms: AbstractTerm) -> int:
        del terms
        return self.reversible_order
