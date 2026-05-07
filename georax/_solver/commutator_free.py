from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, override

import jax.numpy as jnp
import numpy as np
from diffrax import (
    RESULTS,
    AbstractSolver,
    AbstractTerm,
    LocalLinearInterpolation,
)
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax_lowstorage import LowStorageRecurrence
from jaxtyping import Array

from georax._term import GeometricTerm, find_geometric_term, select_chart_for_solver


@dataclass(frozen=True)
class CommutatorFreeTableau:
    c: tuple[float, ...]
    stage_exps: tuple[tuple[np.ndarray, ...], ...]
    final_exps: tuple[np.ndarray, ...]
    embedded_final_exps: tuple[np.ndarray, ...] | None = None

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
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
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
        del t0, t1, y0, args
        select_chart_for_solver(self, find_geometric_term(terms))
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
        exp_rows: tuple[np.ndarray, ...],
        stages: list[Array],
        geometric_term: GeometricTerm,
    ) -> Array:
        y = y_base
        for row in exp_rows:
            coeffs = jnp.zeros_like(stages[0])
            for weight, stage in zip(row, stages, strict=True):
                coeffs = coeffs + jnp.asarray(weight, dtype=stage.dtype) * stage
            y = geometric_term.apply_increment(y, coeffs)
        return y

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
        geometric_term = find_geometric_term(terms)
        if geometric_term.geometry.chart is None:
            select_chart_for_solver(self, geometric_term)
        stages: list[Array] = []

        for c_i, exp_rows in zip(self.tableau.c, self.tableau.stage_exps, strict=True):
            y_stage = self._apply_exp_product(y0, exp_rows, stages, geometric_term)
            t_stage = t1 if c_i == 1.0 else t0 + c_i * dt
            stages.append(terms.prod(terms.vf(t_stage, y_stage, args), control))

        y1 = self._apply_exp_product(
            y0, self.tableau.final_exps, stages, geometric_term
        )

        y_error = None
        if self.tableau.embedded_final_exps is not None:
            y_hat = self._apply_exp_product(
                y0,
                self.tableau.embedded_final_exps,
                stages,
                geometric_term,
            )
            # This ambient subtraction is acceptable for now; a geometry-aware
            # difference may be preferable for manifold error control later.
            y_error = y_hat - y1

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful


class AbstractLowStorageCommutatorFreeSolver(AbstractCommutatorFreeSolver):
    recurrence: ClassVar[LowStorageRecurrence]
    embedded_penultimate_exps: ClassVar[tuple[np.ndarray, ...] | None] = None

    @property
    def _tracks_penultimate(self) -> bool:
        return (
            self.recurrence.penultimate_stage_error
            or self.embedded_penultimate_exps is not None
        )

    def error_order(self, terms: GeometricTerm) -> int | None:
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
        geometric_term = find_geometric_term(terms)
        if geometric_term.geometry.chart is None:
            select_chart_for_solver(self, geometric_term)
        stages: list[Array] = []
        last_stage = self.recurrence.num_stages - 1

        t_stage0 = t1 if self.recurrence.C[0] == 1.0 else t0 + c[0] * dt
        tmp = terms.prod(terms.vf(t_stage0, y0, args), control)
        stages.append(tmp)
        y1 = geometric_term.apply_increment(
            y0, jnp.asarray(b[0], dtype=tmp.dtype) * tmp
        )

        y_penultimate = None
        for stage_index in range(1, self.recurrence.num_stages):
            t_stage = (
                t1
                if self.recurrence.C[stage_index] == 1.0
                else t0 + c[stage_index] * dt
            )
            coeffs = terms.prod(terms.vf(t_stage, y1, args), control)
            stages.append(coeffs)
            tmp = jnp.asarray(a[stage_index - 1], dtype=tmp.dtype) * tmp + coeffs
            if self._tracks_penultimate and stage_index == last_stage:
                y_penultimate = y1
            y1 = geometric_term.apply_increment(
                y1, jnp.asarray(b[stage_index], dtype=tmp.dtype) * tmp
            )

        # Ambient subtraction is acceptable for now; a geometry-aware
        # difference may be preferable for manifold error control later.
        y_error = None
        if self.embedded_penultimate_exps is not None:
            assert y_penultimate is not None, (
                "Embedded penultimate exponentials require at least two stages."
            )
            y_hat = self._apply_exp_product(
                y_penultimate,
                self.embedded_penultimate_exps,
                stages,
                geometric_term,
            )
            y_error = y1 - y_hat
        elif y_penultimate is not None:
            y_error = y1 - y_penultimate

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful
