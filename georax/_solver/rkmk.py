from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax import RESULTS, ODETerm
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solver.base import AbstractSolver, AbstractWrappedSolver
from diffrax._solver.runge_kutta import AbstractERK, ButcherTableau
from diffrax._term import WrapTerm
from jaxtyping import Array

from georax._geometry import LieGroup
from georax._term import GeometricTerm


def _combine(weights: np.ndarray, values: list[Array]) -> Array:
    out = jnp.zeros_like(values[0])
    for weight, value in zip(weights, values, strict=True):
        out = out + jnp.asarray(weight, dtype=value.dtype) * value
    return out


class RKMK(AbstractWrappedSolver):
    solver: AbstractSolver = eqx.field(static=True)
    tableau: ButcherTableau = eqx.field(static=True)
    uses_k_dense_info: bool = eqx.field(static=True)
    has_error_estimate: bool = eqx.field(static=True)

    def __init__(self, solver: AbstractSolver):
        if not isinstance(solver, AbstractERK):
            raise TypeError("RKMK requires a base explicit Runge-Kutta solver.")
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "tableau", solver.tableau)
        object.__setattr__(
            self,
            "uses_k_dense_info",
            self.interpolation_cls is not LocalLinearInterpolation,
        )
        object.__setattr__(
            self,
            "has_error_estimate",
            not np.allclose(np.asarray(solver.tableau.b_error), 0.0),
        )

    @property
    def term_structure(self):  # pyright: ignore
        return GeometricTerm

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    def order(self, terms) -> int | None:
        return self.solver.order(terms)

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

        while isinstance(terms, WrapTerm):
            terms = terms.term
        if not isinstance(terms, GeometricTerm):
            raise TypeError("RKMK requires a GeometricTerm.")
        if not isinstance(terms.geometry, LieGroup):
            raise TypeError(f"RKMK requires a LieGroup in the GeometricTerm. Received {terms.geometry}.")
        if not isinstance(terms.inner, ODETerm):
            raise TypeError("RKMK currently only supports ODETerm inputs.")

        control = jnp.asarray(terms.contr(t0, t1))
        dt = t1 - t0
        stage_algebras: list[Array] = []
        ks: list[Array] = []

        for i in range(self.tableau.num_stages):
            if i == 0:
                omega = None
                y_stage = y0
                t_stage = t0
            else:
                omega = control * _combine(self.tableau.a_lower[i - 1], stage_algebras)
                y_stage = terms.frozen_flow(y0, omega)
                c_i = self.tableau.c[i - 1]
                t_stage = t1 if c_i == 1.0 else t0 + c_i * dt

            vf_stage = terms.vf(t_stage, y_stage, args)
            alg_stage = terms.geometry.to_frame(y_stage, vf_stage)
            if omega is None:
                omega = jnp.zeros_like(alg_stage)
            stage_algebras.append(terms.chart_differential_inv(omega, alg_stage))
            ks.append(terms.prod(vf_stage, control))

        omega_sol = control * _combine(self.tableau.b_sol, stage_algebras)
        y1 = terms.frozen_flow(y0, omega_sol)

        y_error = None
        if self.has_error_estimate:
            omega_embedded = omega_sol - control * _combine(
                self.tableau.b_error, stage_algebras
            )
            y_error = y1 - terms.frozen_flow(y0, omega_embedded)

        dense_info = dict(y0=y0, y1=y1)
        if self.uses_k_dense_info:
            dense_info["k"] = jnp.stack(ks)
        return y1, y_error, dense_info, None, RESULTS.successful
