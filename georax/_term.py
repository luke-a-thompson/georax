from __future__ import annotations

from collections.abc import Callable
from typing import Any, override

import equinox as eqx
import jax.numpy as jnp
from diffrax import AbstractTerm
from diffrax._custom_types import Args, RealScalarLike, Y
from diffrax._term import _Control
from jaxtyping import Array

from georax._geometry import Manifold


class GeometricTerm(AbstractTerm[Array, _Control]):
    """Intrinsic manifold term whose vector field is given in frame coordinates."""

    coeffs_fn: Callable[[RealScalarLike, Array, Args], Array] | None = eqx.field(
        static=True, default=None
    )
    geometry: Manifold
    coeffs_prod_fn: Callable[[RealScalarLike, Array, Args, _Control], Array] | None = (
        eqx.field(static=True, default=None)
    )
    control: Any = None
    control_fn: Callable[..., _Control] | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        coeffs: Callable[[RealScalarLike, Array, Args], Array] | None = None,
        geometry: Manifold | None = None,
        *,
        coeffs_prod: Callable[[RealScalarLike, Array, Args, _Control], Array]
        | None = None,
        control: Any = None,
        control_fn: Callable[..., _Control] | None = None,
    ):
        if geometry is None:
            raise TypeError("GeometricTerm requires a geometry.")
        if coeffs is None and coeffs_prod is None:
            raise TypeError("GeometricTerm requires `coeffs` or `coeffs_prod`.")
        object.__setattr__(self, "coeffs_fn", coeffs)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "coeffs_prod_fn", coeffs_prod)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "control_fn", control_fn)

    @override
    def vf(self, t: RealScalarLike, y: Y, args: Args) -> Array:
        del t, args
        return jnp.zeros_like(y)

    @override
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> _Control:
        if self.control_fn is not None:
            return self.control_fn(t0, t1, **kwargs)
        if self.control is not None:
            return self.control.evaluate(t0, t1, **kwargs)
        return t1 - t0

    @override
    def prod(self, vf: Array, control: _Control) -> Y:
        return vf * control

    @override
    def vf_prod(
        self, t: RealScalarLike, y: Y, args: Args, control: _Control
    ) -> Array:
        del t, args, control
        return jnp.zeros_like(y)

    def coeffs(self, t: RealScalarLike, x: Array, args: Args) -> Array:
        """Return intrinsic frame coefficients at ``(t, x)``."""
        if self.coeffs_fn is None:
            raise TypeError("This GeometricTerm does not define `coeffs`.")
        return self.coeffs_fn(t, x, args)

    def coeffs_prod(
        self, t: RealScalarLike, x: Array, args: Args, control: _Control
    ) -> Array:
        """Return controlled frame-coordinate increments."""
        if self.coeffs_prod_fn is not None:
            return self.coeffs_prod_fn(t, x, args, control)
        return self.coeffs(t, x, args) * control

    def apply_increment(self, x: Array, a: Array) -> Array:
        """Apply one intrinsic frame-coordinate increment via the geometry."""
        return self.geometry.apply_increment(x, a)
