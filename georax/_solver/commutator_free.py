from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, override

import jax.numpy as jnp
import numpy as np
from diffrax import RESULTS, AbstractSolver, AbstractTerm, LocalLinearInterpolation
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._term import WrapTerm
from diffrax_lowstorage import LowStorageRecurrence
from jaxtyping import Array

from georax._term import GeometricTerm


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
    term_structure: ClassVar[type[GeometricTerm]] = GeometricTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )
    tableau: ClassVar[CommutatorFreeTableau]

    @override
    def init(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        del terms, t0, t1, y0, args
        return None

    @override
    def func(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    @staticmethod
    def _unwrap_geometric_term(terms: AbstractTerm) -> GeometricTerm:
        base_term = terms
        while isinstance(base_term, WrapTerm):
            base_term = base_term.term
        if not isinstance(base_term, GeometricTerm):
            raise TypeError("Commutator-free solvers require a GeometricTerm.")
        return base_term

    def _apply_exp_product(
        self,
        y_base: Array,
        exp_rows: tuple[np.ndarray, ...],
        stages: list[Array],
        control: Array,
        geometric_term: GeometricTerm,
    ) -> Array:
        y = y_base
        for row in exp_rows:
            coeffs = jnp.zeros_like(stages[0])
            for weight, stage in zip(row, stages, strict=True):
                coeffs = coeffs + jnp.asarray(weight, dtype=stage.dtype) * stage
            flow_coeffs = jnp.asarray(control, dtype=coeffs.dtype) * coeffs
            y = geometric_term.frozen_flow(y, flow_coeffs)
        return y

    @override
    def step(
        self,
        terms: GeometricTerm,
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
        geometric_term = self._unwrap_geometric_term(terms)
        y0_array = y0
        stages: list[Array] = []

        for c_i, exp_rows in zip(self.tableau.c, self.tableau.stage_exps, strict=True):
            if len(exp_rows) == 0:
                y_stage = y0_array
            else:
                y_stage = self._apply_exp_product(
                    y0_array, exp_rows, stages, control, geometric_term
                )

            t_stage = t1 if c_i == 1.0 else t0 + c_i * dt
            stage_vf = terms.vf(t_stage, y_stage, args)
            stages.append(geometric_term.geometry.to_frame(y_stage, stage_vf))

        y1 = self._apply_exp_product(
            y0_array, self.tableau.final_exps, stages, control, geometric_term
        )

        y_error = None
        if self.tableau.embedded_final_exps is not None:
            y_hat = self._apply_exp_product(
                y0_array,
                self.tableau.embedded_final_exps,
                stages,
                control,
                geometric_term,
            )
            # This ambient subtraction is acceptable for now; a geometry-aware
            # difference may be preferable for manifold error control later.
            y_error = y_hat - y1

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful

    @abstractmethod
    def order(self, terms: GeometricTerm) -> int | None:
        del terms
        """Order of convergence for ODEs."""


class AbstractLowStorageCommutatorFreeSolver(AbstractCommutatorFreeSolver):
    recurrence: ClassVar[LowStorageRecurrence]

    def error_order(self, terms: GeometricTerm) -> int | None:
        if not self.recurrence.penultimate_stage_error:
            return None
        return self.order(terms)

    @override
    def step(
        self,
        terms: GeometricTerm,
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
        control = jnp.asarray(terms.contr(t0, t1))
        geometric_term = self._unwrap_geometric_term(terms)
        y0_array = y0

        t_stage0 = t1 if self.recurrence.C[0] == 1.0 else t0 + c[0] * dt
        tmp = control * geometric_term.coeffs(t_stage0, y0_array, args)
        y1 = geometric_term.frozen_flow(
            y0_array, jnp.asarray(b[0], dtype=tmp.dtype) * tmp
        )

        y_penultimate = None
        for stage_index in range(1, self.recurrence.num_stages):
            t_stage = (
                t1
                if self.recurrence.C[stage_index] == 1.0
                else t0 + c[stage_index] * dt
            )
            coeffs = control * geometric_term.coeffs(t_stage, y1, args)
            tmp = jnp.asarray(a[stage_index - 1], dtype=tmp.dtype) * tmp + coeffs
            if (
                self.recurrence.penultimate_stage_error
                and stage_index == self.recurrence.num_stages - 1
            ):
                y_penultimate = y1
            y1 = geometric_term.frozen_flow(
                y1, jnp.asarray(b[stage_index], dtype=tmp.dtype) * tmp
            )

        y_error = None
        if y_penultimate is not None:
            # This ambient subtraction is acceptable for now; a geometry-aware
            # difference may be preferable for manifold error control later.
            y_error = y1 - y_penultimate

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful
