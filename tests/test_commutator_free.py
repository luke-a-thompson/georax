from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import EuclideanOps
from diffrax import ODETerm
from diffrax_lowstorage import LowStorageRecurrence
from jaxtyping import Array

from georax import (
    AbstractCommutatorFreeSolver,
    AbstractLowStorageCommutatorFreeSolver,
)
from georax._geometry import Manifold
from georax._solver.commutator_free import CommutatorFreeTableau
from georax._term import GeometricTerm


class AffineRetractionOps(Manifold):
    scale: float

    def frame(self, x: Array) -> Array:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    def retraction(self, x: Array, v: Array) -> Array:
        return self.scale * x + v


class TableauSolver(AbstractCommutatorFreeSolver):
    tableau: ClassVar[CommutatorFreeTableau]
    _order: ClassVar[int]

    def order(self, terms: GeometricTerm) -> int | None:
        del terms
        return self._order


class LowStorageSolver(AbstractLowStorageCommutatorFreeSolver):
    recurrence: ClassVar[LowStorageRecurrence]
    _order: ClassVar[int]

    def order(self, terms: GeometricTerm) -> int | None:
        del terms
        return self._order


def _solver_type(name: str, tableau: CommutatorFreeTableau, order: int):
    return type(name, (TableauSolver,), {"tableau": tableau, "_order": order})


def _low_storage_solver_type(name: str, recurrence: LowStorageRecurrence, order: int):
    return type(name, (LowStorageSolver,), {"recurrence": recurrence, "_order": order})


def _make_term(vf, geometry: Manifold) -> GeometricTerm:
    return GeometricTerm(inner=ODETerm(vf), geometry=geometry)


def _run_step(
    solver: AbstractCommutatorFreeSolver,
    terms,
    y0: Array = jnp.array([1.0]),
    t0: float = 0.0,
    t1: float = 1.0,
):
    y1, y_error, dense_info, solver_state, result = solver.step(
        terms=terms,
        t0=t0,
        t1=t1,
        y0=y0,
        args=None,
        solver_state=None,
        made_jump=False,
    )
    return y1, y_error, dense_info, solver_state, result


def test_tableau_shape_validation() -> None:
    with pytest.raises(ValueError, match="`c` and `stage_exps`"):
        CommutatorFreeTableau(
            c=(0.0,),
            stage_exps=((), (np.array([1.0]),)),
            final_exps=(np.array([1.0]),),
        )

    with pytest.raises(ValueError, match="Stage exponential coefficients"):
        CommutatorFreeTableau(
            c=(0.0, 0.5),
            stage_exps=((), (np.array([1.0, 2.0]),)),
            final_exps=(np.array([0.0, 1.0]),),
        )

    with pytest.raises(ValueError, match="Final exponential coefficients"):
        CommutatorFreeTableau(
            c=(0.0,),
            stage_exps=((),),
            final_exps=(np.array([1.0, 2.0]),),
        )

    with pytest.raises(ValueError, match="Embedded final exponential coefficients"):
        CommutatorFreeTableau(
            c=(0.0,),
            stage_exps=((),),
            final_exps=(np.array([1.0]),),
            embedded_final_exps=(np.array([1.0, 2.0]),),
        )


def test_multi_exponential_stage_applies_rows_in_order() -> None:
    tableau = CommutatorFreeTableau(
        c=(0.0, 1.0),
        stage_exps=(
            (),
            (np.array([1.0]), np.array([2.0])),
        ),
        final_exps=(np.array([0.0, 1.0]),),
    )
    StageProductSolver = _solver_type("StageProductSolver", tableau, 1)

    def vf(t, y, args):
        del t, args
        return y

    y1, _, _, _, _ = _run_step(
        StageProductSolver(),
        _make_term(vf, AffineRetractionOps(scale=10.0)),
    )

    assert bool(jnp.allclose(y1, jnp.array([122.0])))


def test_final_multi_exponential_product_applies_all_rows() -> None:
    tableau = CommutatorFreeTableau(
        c=(0.0,),
        stage_exps=((),),
        final_exps=(np.array([1.0]), np.array([2.0])),
    )
    FinalProductSolver = _solver_type("FinalProductSolver", tableau, 1)

    def vf(t, y, args):
        del t, y, args
        return jnp.array([3.0])

    y1, _, _, _, _ = _run_step(
        FinalProductSolver(),
        _make_term(vf, AffineRetractionOps(scale=10.0)),
    )

    assert bool(jnp.allclose(y1, jnp.array([136.0])))


def test_embedded_pair_returns_error_estimate() -> None:
    tableau = CommutatorFreeTableau(
        c=(0.0,),
        stage_exps=((),),
        final_exps=(np.array([1.0]),),
        embedded_final_exps=(np.array([2.0]),),
    )
    EmbeddedSolver = _solver_type("EmbeddedSolver", tableau, 1)

    def vf(t, y, args):
        del t, y, args
        return jnp.array([3.0])

    y1, y_error, dense_info, _, _ = _run_step(
        EmbeddedSolver(),
        _make_term(vf, EuclideanOps()),
    )

    assert bool(jnp.allclose(y1, jnp.array([4.0])))
    assert y_error is not None
    assert bool(jnp.allclose(y_error, jnp.array([3.0])))
    assert bool(jnp.allclose(dense_info["y0"], jnp.array([1.0])))
    assert bool(jnp.allclose(dense_info["y1"], y1))


def test_low_storage_recurrence_applies_chained_flows() -> None:
    recurrence = LowStorageRecurrence(
        A=np.array([0.5]),
        B=np.array([0.5, 1.0]),
        C=np.array([0.0, 1.0]),
    )
    TwoStageLowStorageSolver = _low_storage_solver_type(
        "TwoStageLowStorageSolver", recurrence, 2
    )

    def vf(t, y, args):
        del t, args
        return y

    y1, y_error, _, _, _ = _run_step(
        TwoStageLowStorageSolver(),
        _make_term(vf, EuclideanOps()),
    )

    assert y_error is None
    assert bool(jnp.allclose(y1, jnp.array([3.5])))
