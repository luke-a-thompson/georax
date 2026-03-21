from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import ODETerm
from diffrax._term import WrapTerm
from jaxtyping import Array

from georax._geometry import GeometricOps
from georax._term import GeometricTerm
from georax.base import AbstractCommutatorFreeSolver, CommutatorFreeTableau
from georax.cg2 import CG2


class EuclideanOps(GeometricOps):
    def frame(self, x: Array) -> Array:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    def retraction(self, x: Array, v: Array) -> Array:
        return x + v


class AffineRetractionOps(GeometricOps):
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


def _solver_type(name: str, tableau: CommutatorFreeTableau, order: int):
    return type(name, (TableauSolver,), {"tableau": tableau, "_order": order})


def _make_term(vf, geometry: GeometricOps) -> GeometricTerm:
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


def _old_single_exp_step(
    c: tuple[float, ...],
    a: tuple[tuple[float, ...], ...],
    b: tuple[float, ...],
    vf,
    y0: Array,
    t0: float,
    t1: float,
) -> Array:
    dt = t1 - t0
    stages: list[Array] = []

    for c_i, a_i in zip(c, a, strict=True):
        if len(a_i) == 0:
            y_stage = y0
        else:
            stage_coeffs = jnp.zeros_like(stages[0])
            for weight, stage in zip(a_i, stages, strict=True):
                stage_coeffs = stage_coeffs + weight * stage
            y_stage = y0 + stage_coeffs

        t_stage = t1 if c_i == 1.0 else t0 + c_i * dt
        stages.append(vf(t_stage, y_stage, None))

    output_coeffs = jnp.zeros_like(stages[0])
    for weight, stage in zip(b, stages, strict=True):
        output_coeffs = output_coeffs + weight * stage
    return y0 + output_coeffs


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


def test_single_exponential_tableau_matches_old_narrow_step() -> None:
    tableau = CommutatorFreeTableau(
        c=(0.0, 0.5, 1.0),
        stage_exps=(
            (),
            (np.array([0.5]),),
            (np.array([0.25, 0.75]),),
        ),
        final_exps=(np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),),
    )
    ThreeStageSolver = _solver_type("ThreeStageSolver", tableau, 3)

    def vf(t, y, args):
        del args
        return jnp.array([2.0 * t + y[0]])

    y0 = jnp.array([1.0])
    expected = _old_single_exp_step(
        c=(0.0, 0.5, 1.0),
        a=((), (0.5,), (0.25, 0.75)),
        b=(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        vf=vf,
        y0=y0,
        t0=0.0,
        t1=1.0,
    )
    y1, y_error, _, _, _ = _run_step(ThreeStageSolver(), _make_term(vf, EuclideanOps()))

    assert y_error is None
    assert bool(jnp.allclose(y1, expected))


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


def test_wrapped_geometric_term_is_accepted() -> None:
    def vf(t, y, args):
        del t, args
        return 2.0 * y

    term = _make_term(vf, EuclideanOps())
    wrapped_term = WrapTerm(term=term, direction=1)

    y1, y_error, _, _, _ = _run_step(CG2(), wrapped_term)

    assert y_error is None
    assert bool(jnp.allclose(y1, jnp.array([5.0])))


def test_euclidean_consistency_matches_additive_composition() -> None:
    tableau = CommutatorFreeTableau(
        c=(0.0, 1.0),
        stage_exps=(
            (),
            (np.array([0.25]), np.array([0.75])),
        ),
        final_exps=(np.array([0.5, 0.0]), np.array([0.0, 1.0])),
    )
    EuclideanSolver = _solver_type("EuclideanSolver", tableau, 1)

    def vf(t, y, args):
        del args
        return jnp.array([t + y[0]])

    y1, _, _, _, _ = _run_step(EuclideanSolver(), _make_term(vf, EuclideanOps()))

    assert bool(jnp.allclose(y1, jnp.array([4.5])))
