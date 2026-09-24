from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Literal, override

import equinox as eqx
from diffrax import (
    RESULTS,
    AbstractBrownianIncrement,
    AbstractSRK,
    AbstractTerm,
    AbstractWrappedSolver,
    MultiTerm,
    StochasticButcherTableau,
)

from georax._compat import (
    VF,
    AdditiveCoeffs,
    Args,
    BoolScalarLike,
    DenseInfo,
    GeneralCoeffs,
    RealScalarLike,
    WrapTerm,
    Y,
)
from georax._term import (
    GeometricTerm,
    Pullback,
    PulledDiffusionTerm,
    PulledDriftTerm,
    find_geometry,
    select_chart_for_solver,
    unwrap_term,
)

from ._interpolation import GeometricInterpolation
from ._pullback import pullback_step

SRKTerms = MultiTerm[
    tuple[AbstractTerm[VF, RealScalarLike], AbstractTerm[VF, AbstractBrownianIncrement]]
]


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

    ??? Reference

        ```bibtex
        @article{MunizEhrhardtGuentherWinkler2023,
          title = {Strong stochastic Runge-Kutta-Munthe-Kaas methods for nonlinear Ito SDEs on manifolds},
          author = {Muniz, Michelle and Ehrhardt, Matthias and Guenther, Michael and Winkler, Renate},
          journal = {Applied Numerical Mathematics},
          volume = {193},
          pages = {196--203},
          year = {2023},
          doi = {10.1016/j.apnum.2023.07.024}
        }
        ```
    """

    # Equinox freezes this field; it implements Diffrax's AbstractVar.
    solver: AbstractSRK = eqx.field(static=True)  # pyright: ignore[reportIncompatibleVariableOverride]
    term_structure: ClassVar[type[MultiTerm[tuple[GeometricTerm, AbstractTerm]]]] = (
        MultiTerm[tuple[GeometricTerm, AbstractTerm]]
    )
    interpolation_cls: ClassVar[Callable[..., GeometricInterpolation]] = (
        GeometricInterpolation
    )
    term_compatible_contr_kwargs: ClassVar[tuple[dict[str, bool], dict[str, bool]]] = (
        {},
        {"use_levy": True},
    )
    is_additive: bool = eqx.field(static=True)
    additive_after_pullback: bool = eqx.field(static=True)

    def __init__(
        self,
        solver: AbstractSRK,
        *,
        additive_after_pullback: bool = False,
    ) -> None:
        if not isinstance(solver, AbstractSRK):
            raise TypeError("SRKMK requires a base stochastic Runge--Kutta solver.")

        # A fixed-length lax scan supports both differentiation modes. Diffrax's
        # adjoints inspect scan_kind on wrappers, so expose the selected setting.
        if solver.scan_kind is None:
            solver = eqx.tree_at(
                lambda s: s.scan_kind, solver, "lax", is_leaf=lambda x: x is None
            )

        tableau = solver.tableau
        coeffs = (
            tableau.coeffs_w,
            tableau.coeffs_hh,
            tableau.coeffs_kk,
        )

        uses_additive = any(isinstance(coeff, AdditiveCoeffs) for coeff in coeffs)
        uses_general = any(isinstance(coeff, GeneralCoeffs) for coeff in coeffs)
        if uses_additive and uses_general:
            raise TypeError(
                "SRKMK currently expects either all additive SRK coefficients "
                "or all general SRK coefficients, not a mixture."
            )

        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "is_additive", uses_additive)
        object.__setattr__(self, "additive_after_pullback", additive_after_pullback)

    @property
    def scan_kind(self) -> Literal["lax", "checkpointed"] | None:
        return self.solver.scan_kind

    @property
    def tableau(self) -> StochasticButcherTableau:
        return self.solver.tableau

    @property
    def minimal_levy_area(self) -> type[AbstractBrownianIncrement]:
        return self.solver.minimal_levy_area

    def order(self, terms: AbstractTerm) -> int | None:
        return self.solver.order(terms)

    def strong_order(self, terms: AbstractTerm) -> RealScalarLike | None:
        return self.solver.strong_order(terms)

    @staticmethod
    def _split_terms(terms: AbstractTerm) -> tuple[AbstractTerm, AbstractTerm]:
        if isinstance(terms, WrapTerm):
            drift, diffusion = SRKMK._split_terms(terms.term)
            return WrapTerm(drift, terms.direction), WrapTerm(
                diffusion, terms.direction
            )
        if not isinstance(terms, MultiTerm) or len(terms.terms) != 2:
            raise TypeError(
                "SRKMK expects terms = MultiTerm(drift_term, diffusion_term)."
            )
        drift_term, diffusion_term = terms.terms
        if not isinstance(unwrap_term(drift_term), GeometricTerm):
            raise TypeError("SRKMK requires a geometric drift term.")
        return drift_term, diffusion_term

    @override
    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> None:
        if self.is_additive and not self.additive_after_pullback:
            raise TypeError(
                "This SRK tableau assumes additive noise. For SRKMK this means "
                "the pulled-back Lie-algebra diffusion must be independent of "
                "the algebra state. Pass additive_after_pullback=True only if "
                "you have ensured this."
            )

        pullback, algebra_terms = self._local_terms(terms, y0)
        self.solver.init(algebra_terms, t0, t1, pullback.zero(), args)
        return None

    def _local_terms(
        self, terms: AbstractTerm, y0: Y
    ) -> tuple[Pullback[Any], SRKTerms]:
        drift, diffusion = self._split_terms(terms)
        geometry = find_geometry(drift)
        chart = select_chart_for_solver(self, terms, geometry, pullback=True)
        pullback = Pullback(geometry, chart, y0)
        return pullback, MultiTerm(
            PulledDriftTerm(drift, pullback),
            PulledDiffusionTerm(diffusion, pullback),
        )

    @override
    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        drift_term, _ = self._split_terms(terms)
        return drift_term.vf(t0, y0, args)

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
        del solver_state

        pullback, algebra_terms = self._local_terms(terms, y0)
        return pullback_step(
            self.solver, algebra_terms, pullback, t0, t1, args, made_jump
        )
