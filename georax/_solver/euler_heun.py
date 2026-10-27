from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, override

from diffrax import RESULTS, AbstractStratonovichSolver, AbstractTerm, MultiTerm

from georax._compat import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from georax._term import GeometricTerm, find_geometry

from ._interpolation import GeometricInterpolation, geometric_dense_info


class GeometricEulerHeun(AbstractStratonovichSolver):
    """Geometric Euler--Heun method for Stratonovich SDEs.

    Use ``MultiTerm(GeometricTerm(drift, geometry), diffusion_term)`` with both
    vector fields returning frame-coordinate coefficients. A noise-only chart
    predictor supplies the second diffusion evaluation at the same time. The
    averaged diffusion increment and left-endpoint drift are applied together
    through an order-2 chart. On Euclidean geometry this is Diffrax's Euler--Heun
    method.

    The method has ODE order 1 and strong order 0.5 for general noise. It has no
    embedded error estimate. Saved intermediate states use chart interpolation.
    """

    term_structure: ClassVar[type[MultiTerm[tuple[GeometricTerm, AbstractTerm]]]] = (
        MultiTerm[tuple[GeometricTerm, AbstractTerm]]
    )
    interpolation_cls: ClassVar[Callable[..., GeometricInterpolation]] = (
        GeometricInterpolation
    )

    def order(self, terms: AbstractTerm) -> int:
        return 1

    def strong_order(self, terms: AbstractTerm) -> float:
        return 0.5

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

    @override
    def func(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)

    @override
    def step(
        self,
        terms: MultiTerm[tuple[AbstractTerm, AbstractTerm]],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, None, DenseInfo, None, RESULTS]:
        del solver_state, made_jump

        geometry = find_geometry(terms)
        chart = geometry.select_chart(2)
        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        dw = diffusion.contr(t0, t1)

        # ControlTerm.vf_prod checks against the ambient state shape, whereas
        # these products live in the geometry's frame-coordinate shape.
        f0 = drift.prod(drift.vf(t0, y0, args), dt)
        g0 = diffusion.prod(diffusion.vf(t0, y0, args), dw)
        predictor = geometry.apply_increment(y0, g0, chart)
        g1 = diffusion.prod(diffusion.vf(t0, predictor, args), dw)
        increment = f0 + 0.5 * (g0 + g1)
        y1 = geometry.apply_increment(y0, increment, chart)

        dense_info = geometric_dense_info(y0, y1, (increment,), geometry, chart)
        return y1, None, dense_info, None, RESULTS.successful
