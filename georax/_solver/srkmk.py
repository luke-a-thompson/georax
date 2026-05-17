from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
from diffrax import RESULTS, AbstractTerm, MultiTerm
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._local_interpolation import LocalLinearInterpolation
from diffrax._solver.base import AbstractSolver, AbstractWrappedSolver
from diffrax._solver.srk import (
    AbstractSRK,
    AdditiveCoeffs,
    GeneralCoeffs,
    StochasticButcherTableau,
)

from georax._term import (
    GeometricTerm,
    PulledDiffusionTerm,
    PulledDriftTerm,
    select_chart_for_solver,
    unwrap_term,
)


class SRKMK(AbstractWrappedSolver):
    """RKMK lift of a Diffrax stochastic Runge--Kutta solver.

    The first term is a ``GeometricTerm`` for the drift in frame coordinates. The
    second term is a controlled diffusion term whose vector field also returns
    frame-coordinate coefficients. It is pulled back through the selected local
    chart at each stage.

    This wrapper preserves the stochastic interpretation of the wrapped SRK.
    Diffrax's current ``AbstractSRK`` methods are Stratonovich methods; Itô
    Magnus drift corrections must be included in the supplied drift term or in a
    dedicated Itô SRK implementation.
    """

    solver: AbstractSolver = eqx.field(static=True)
    tableau: StochasticButcherTableau = eqx.field(static=True)
    is_additive: bool = eqx.field(static=True)
    additive_after_pullback: bool = eqx.field(static=True)

    def __init__(
        self,
        solver: AbstractSolver,
        *,
        additive_after_pullback: bool = False,
    ):
        if not isinstance(solver, AbstractSRK):
            raise TypeError("SRKMK requires a base stochastic Runge--Kutta solver.")

        tableau = solver.tableau
        coeffs = [
            tableau.coeffs_w,
            tableau.coeffs_hh,
            tableau.coeffs_kk,
        ]
        coeffs = [coeff for coeff in coeffs if coeff is not None]

        uses_additive = any(isinstance(coeff, AdditiveCoeffs) for coeff in coeffs)
        uses_general = any(isinstance(coeff, GeneralCoeffs) for coeff in coeffs)
        if uses_additive and uses_general:
            raise TypeError(
                "SRKMK currently expects either all additive SRK coefficients "
                "or all general SRK coefficients, not a mixture."
            )

        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "tableau", tableau)
        object.__setattr__(self, "is_additive", uses_additive)
        object.__setattr__(self, "additive_after_pullback", additive_after_pullback)

    @property
    def term_structure(self):  # pyright: ignore
        return MultiTerm[tuple[GeometricTerm, AbstractTerm]]

    @property
    def term_compatible_contr_kwargs(self):  # pyright: ignore
        return (dict(), dict(use_levy=True))

    @property
    def interpolation_cls(self):  # pyright: ignore
        # Keep this conservative. SRKMK dense output should not inherit any
        # Euclidean RK Hermite interpolation without geometric reconstruction.
        return LocalLinearInterpolation

    @property
    def minimal_levy_area(self):  # pyright: ignore
        return self.solver.minimal_levy_area

    def order(self, terms) -> int | None:
        return self.solver.order(terms)

    def strong_order(self, terms):
        return self.solver.strong_order(terms)

    @staticmethod
    def _split_terms(terms) -> tuple[GeometricTerm, AbstractTerm]:
        terms = unwrap_term(terms)
        if not isinstance(terms, MultiTerm) or len(terms.terms) != 2:
            raise TypeError(
                "SRKMK expects terms = MultiTerm(drift_term, diffusion_term)."
            )
        drift_term, diffusion_term = terms.terms
        drift_term = unwrap_term(drift_term)
        if not isinstance(drift_term, GeometricTerm):
            raise TypeError("SRKMK requires a geometric drift term.")
        return drift_term, diffusion_term

    @override
    def init(
        self,
        terms,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        drift_term, diffusion_term = self._split_terms(terms)

        if self.is_additive and not self.additive_after_pullback:
            raise TypeError(
                "This SRK tableau assumes additive noise. For SRKMK this means "
                "the pulled-back Lie-algebra diffusion must be independent of "
                "the algebra state. Pass additive_after_pullback=True only if "
                "you have ensured this."
            )

        # Chart selection is delegated to the geometry. For exponential-style
        # charts this should be conservative enough to satisfy the SRKMK
        # truncation condition for the wrapped method.
        select_chart_for_solver(self, drift_term.geometry)

        # Defer to the wrapped SRK's init for any trace-time validation. For
        # additive tableaus this performs a JVP check that the (pulled-back)
        # diffusion is independent of the algebra state — i.e. it catches a
        # false ``additive_after_pullback=True`` claim. For general tableaus
        # AbstractSRK.init is a no-op.
        omega0 = jnp.zeros(
            drift_term.geometry.coordinate_shape, dtype=jnp.result_type(y0)
        )
        algebra_terms = MultiTerm(
            PulledDriftTerm(drift_term, y0),
            PulledDiffusionTerm(drift_term, diffusion_term, y0, omega0),
        )
        self.solver.init(algebra_terms, t0, t1, omega0, args)
        return None

    @override
    def func(
        self,
        terms,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift_term, _ = self._split_terms(terms)
        return drift_term.vf(t0, y0, args)

    @override
    def step(
        self,
        terms,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y | None, DenseInfo, None, RESULTS]:
        del solver_state

        drift_term, diffusion_term = self._split_terms(terms)
        geometry = drift_term.geometry
        chart = geometry.chart
        if chart is None:
            raise TypeError("SRKMK requires a geometry with a selected chart.")

        omega0 = jnp.zeros(geometry.coordinate_shape, dtype=jnp.result_type(y0))
        algebra_terms = MultiTerm(
            PulledDriftTerm(drift_term, y0),
            PulledDiffusionTerm(drift_term, diffusion_term, y0, omega0),
        )

        omega1, omega_error, _, _, result = self.solver.step(
            algebra_terms,
            t0,
            t1,
            omega0,
            args,
            None,
            made_jump,
        )
        y1 = geometry.apply_increment(y0, omega1)

        y_error = None
        if omega_error is not None:
            y_error = geometry.apply_increment(y0, omega_error) - y0

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, result
