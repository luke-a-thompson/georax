from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax.numpy as jnp
import numpy as np
from diffrax import (
    RESULTS,
    AbstractReversibleSolver,
    AbstractSolver,
    AbstractStratonovichSolver,
    AbstractTerm,
)
from diffrax_lowstorage import LowStorageRecurrence
from jaxtyping import Array, PyTree
from numpy.typing import NDArray

from georax._compat import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from georax._geometry import LocalChart, Manifold
from georax._term import find_geometry, select_chart_for_solver

from ._interpolation import GeometricInterpolation, geometric_dense_info


@dataclass(frozen=True)
class CommutatorFreeTableau:
    c: tuple[float, ...]
    stage_exps: tuple[tuple[NDArray[np.float64], ...], ...]
    final_exps: tuple[NDArray[np.float64], ...]
    embedded_final_exps: tuple[NDArray[np.float64], ...] | None = None

    def __post_init__(self) -> None:
        num_stages = len(self.c)
        if num_stages == 0:
            raise ValueError(
                "Commutator-free tableaus must contain at least one stage."
            )
        if len(self.stage_exps) != num_stages:
            raise ValueError("`c` and `stage_exps` must have the same length.")

        for stage_index, exp_rows in enumerate(self.stage_exps):
            for exp_index, row in enumerate(exp_rows):
                if row.ndim != 1 or row.shape != (stage_index,):
                    raise ValueError(
                        "Stage exponential coefficients must be one-dimensional "
                        f"arrays of shape ({stage_index},); got shape {row.shape} "
                        f"for stage {stage_index} exponential {exp_index}."
                    )

        for exp_index, row in enumerate(self.final_exps):
            if row.ndim != 1 or row.shape != (num_stages,):
                raise ValueError(
                    "Final exponential coefficients must be one-dimensional arrays "
                    f"of shape ({num_stages},); got shape {row.shape} for final "
                    f"exponential {exp_index}."
                )

        if self.embedded_final_exps is not None:
            for exp_index, row in enumerate(self.embedded_final_exps):
                if row.ndim != 1 or row.shape != (num_stages,):
                    raise ValueError(
                        "Embedded final exponential coefficients must be "
                        "one-dimensional arrays of shape "
                        f"({num_stages},); got shape {row.shape} for embedded "
                        f"final exponential {exp_index}."
                    )


class AbstractCommutatorFreeSolver(AbstractSolver):
    term_structure: ClassVar[type[AbstractTerm]] = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., GeometricInterpolation]] = (
        GeometricInterpolation
    )
    tableau: ClassVar[CommutatorFreeTableau]

    @override
    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        del t0, t1, args
        find_geometry(terms).check_state_shape(y0)
        return None

    @override
    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    def _apply_exp_product(
        self,
        y_base: Array,
        exp_rows: tuple[NDArray[np.float64], ...],
        stages: list[Array],
        geometry: Manifold[Any],
        chart: LocalChart[Any],
    ) -> Array:
        y = y_base
        for coeffs in self._increments(exp_rows, stages):
            y = geometry.apply_increment(y, coeffs, chart)
        return y

    @staticmethod
    def _increments(
        exp_rows: tuple[NDArray[np.float64], ...], stages: list[Array]
    ) -> tuple[Array, ...]:
        return tuple(
            sum(
                (
                    jnp.asarray(weight, dtype=stage.dtype) * stage
                    for weight, stage in zip(row, stages, strict=True)
                ),
                jnp.zeros_like(stages[0]),
            )
            for row in exp_rows
        )

    @override
    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, None, RESULTS]:
        del solver_state, made_jump

        dt = t1 - t0
        control = terms.contr(t0, t1)
        geometry = find_geometry(terms)
        chart = select_chart_for_solver(self, terms, geometry)
        stages: list[Array] = []

        for c_i, exp_rows in zip(self.tableau.c, self.tableau.stage_exps, strict=True):
            y_stage = self._apply_exp_product(y0, exp_rows, stages, geometry, chart)
            t_stage = t1 if c_i == 1.0 else t0 + c_i * dt
            stages.append(terms.prod(terms.vf(t_stage, y_stage, args), control))

        y1 = self._apply_exp_product(
            y0, self.tableau.final_exps, stages, geometry, chart
        )

        y_error = None
        if self.tableau.embedded_final_exps is not None:
            y_hat = self._apply_exp_product(
                y0,
                self.tableau.embedded_final_exps,
                stages,
                geometry,
                chart,
            )
            # This ambient subtraction is acceptable for now; a geometry-aware
            # difference may be preferable for manifold error control later.
            y_error = y1 - y_hat

        dense_info = geometric_dense_info(
            y0, y1, self._increments(self.tableau.final_exps, stages), geometry, chart
        )
        return y1, y_error, dense_info, None, RESULTS.successful


class AbstractLowStorageCommutatorFreeSolver(AbstractCommutatorFreeSolver):
    recurrence: ClassVar[LowStorageRecurrence]
    embedded_penultimate_exps: ClassVar[tuple[NDArray[np.float64], ...] | None] = None
    embedded_final_increment: ClassVar[bool] = False

    @property
    def _tracks_penultimate(self) -> bool:
        return (
            self.recurrence.penultimate_stage_error
            or self.embedded_penultimate_exps is not None
        )

    def error_order(self, terms: AbstractTerm) -> RealScalarLike | None:
        if self.embedded_final_increment:
            # A 2(1) pair for ODEs; use Diffrax's strong-order convention for SDEs.
            return AbstractSolver.error_order(self, terms)
        return self.order(terms) if self._tracks_penultimate else None

    @override
    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, None, RESULTS]:
        del solver_state, made_jump

        a = jnp.asarray(self.recurrence.A)
        b = jnp.asarray(self.recurrence.B)
        c = jnp.asarray(self.recurrence.C)

        dt = t1 - t0
        control = terms.contr(t0, t1)
        geometry = find_geometry(terms)
        chart = select_chart_for_solver(self, terms, geometry)
        stages: list[Array] | None = (
            [] if self.embedded_penultimate_exps is not None else None
        )
        last_stage = self.recurrence.num_stages - 1

        t_stage0 = t1 if self.recurrence.C[0] == 1.0 else t0 + c[0] * dt
        tmp = terms.prod(terms.vf(t_stage0, y0, args), control)
        if stages is not None:
            stages.append(tmp)
        increment = jnp.asarray(b[0], dtype=tmp.dtype) * tmp
        increments = [increment]
        y1 = geometry.apply_increment(y0, increment, chart)

        y_penultimate = None
        for stage_index in range(1, self.recurrence.num_stages):
            t_stage = (
                t1
                if self.recurrence.C[stage_index] == 1.0
                else t0 + c[stage_index] * dt
            )
            coeffs = terms.prod(terms.vf(t_stage, y1, args), control)
            if stages is not None:
                stages.append(coeffs)
            tmp = jnp.asarray(a[stage_index - 1], dtype=tmp.dtype) * tmp + coeffs
            if self._tracks_penultimate and stage_index == last_stage:
                y_penultimate = y1
            increment = jnp.asarray(b[stage_index], dtype=tmp.dtype) * tmp
            increments.append(increment)
            y1 = geometry.apply_increment(y1, increment, chart)

        # Ambient subtraction is acceptable for now; a geometry-aware
        # difference may be preferable for manifold error control later.
        y_error = None
        if self.embedded_final_increment:
            # A omits the first (zero) coefficient, so d_1 = 1.
            # Compute this scalar from the static coefficients, without retaining
            # generator evaluations or another algebra accumulator.
            normalization = 1.0
            for a_i in self.recurrence.A:
                normalization = 1.0 + float(a_i) * normalization
            if normalization == 0.0:
                raise ValueError(
                    "An embedded final increment requires nonzero normalization."
                )
            y_hat = geometry.apply_increment(
                y0, tmp / jnp.asarray(normalization, dtype=tmp.dtype), chart
            )
            y_error = y1 - y_hat
        elif self.embedded_penultimate_exps is not None:
            assert y_penultimate is not None, (
                "Embedded penultimate exponentials require at least two stages."
            )
            assert stages is not None
            y_hat = self._apply_exp_product(
                y_penultimate,
                self.embedded_penultimate_exps,
                stages,
                geometry,
                chart,
            )
            y_error = y1 - y_hat
        elif y_penultimate is not None:
            y_error = y1 - y_penultimate

        dense_info = geometric_dense_info(y0, y1, increments, geometry, chart)
        return y1, y_error, dense_info, None, RESULTS.successful


class _AbstractCFEES(
    AbstractLowStorageCommutatorFreeSolver,
    AbstractReversibleSolver,
    AbstractStratonovichSolver,
):
    """Shared reversible solver state and orders for the CF-EES family."""

    embedded_final_increment: ClassVar[bool] = True

    @override
    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> Y:
        super().init(terms, t0, t1, y0, args)
        return y0

    @override
    def order(self, terms: AbstractTerm) -> int:
        del terms
        return 2

    def strong_order(self, terms: AbstractTerm) -> float:
        del terms
        return 0.5

    @override
    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: Y,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, Y, RESULTS]:
        del solver_state
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
        solver_state: Y,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, DenseInfo, Y, RESULTS]:
        del ts_state
        y0, _, dense_info, solver_state, result = self.step(
            terms, t1, t0, y1, args, solver_state, made_jump
        )
        return y0, dense_info, solver_state, result
