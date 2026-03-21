from __future__ import annotations

from typing import override

from jaxtyping import Array

from diffrax._custom_types import Args, RealScalarLike, Y
from diffrax._term import AbstractTerm, _Control, _VF

from ._geometry import GeometricOps


class GeometricTerm(AbstractTerm[_VF, _Control]):
    """Wraps an AbstractTerm with manifold geometry for Crouch--Grossman integrators.

    Delegates ``vf``/``contr``/``prod`` to an inner term, and adds the
    frame-coordinate methods needed by manifold Runge--Kutta schemes.
    """

    inner: AbstractTerm[_VF, _Control]
    geometry: GeometricOps

    @override
    def vf(self, t: RealScalarLike, y: Y, args: Args) -> _VF:
        return self.inner.vf(t, y, args)

    @override
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        return self.inner.contr(t0, t1, **kwargs)

    @override
    def prod(self, vf: _VF, control: _Control) -> Y:
        return self.inner.prod(vf, control)

    @override
    def vf_prod(self, t: RealScalarLike, y: Y, args: Args, control: _Control) -> Y:
        return self.inner.vf_prod(t, y, args, control)

    def coeffs(self, t: RealScalarLike, x: Array, args: Args) -> Array:
        """Return frame coefficients a such that vf(t, x, args) = sum_d a_d E_d(x)."""
        v = self.inner.vf(t, x, args)
        return self.geometry.to_frame(x, v)

    def coeffs_prod(
        self, t: RealScalarLike, x: Array, args: Args, control: _Control
    ) -> Array:
        """Return frame coefficients of the controlled tangent increment."""
        v = self.inner.vf_prod(t, x, args, control)
        return self.geometry.to_frame(x, v)

    def tangent_from_coeffs(self, x: Array, a: Array) -> Array:
        """Return the ambient tangent vector sum_d a_d E_d(x)."""
        return self.geometry.from_frame(x, a)

    def frozen_flow(self, x: Array, a: Array) -> Array:
        """Apply one frozen subflow at x via the manifold retraction.

        Approximates exp(sum_d a_d E_d) . x by retracting sum_d a_d E_d(x).
        """
        v = self.geometry.from_frame(x, a)
        return self.geometry.retraction(x, v)
