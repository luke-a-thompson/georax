from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from diffrax import AbstractLocalInterpolation
from jaxtyping import Array

from georax._compat import DenseInfo, RealScalarLike
from georax._geometry import LocalChart, Manifold


class GeometricInterpolation(AbstractLocalInterpolation):
    """Continuous chart path through the accepted step's finite motions.

    This preserves the geometry but is not a high-order continuous extension
    of the wrapped method. An interval evaluation returns an ambient difference.
    """

    # Equinox fields implement Diffrax's AbstractVar declarations.
    t0: RealScalarLike  # pyright: ignore[reportIncompatibleVariableOverride]
    t1: RealScalarLike  # pyright: ignore[reportIncompatibleVariableOverride]
    y0: Array
    y1: Array
    increments: tuple[Array, ...]
    geometry: Manifold[Any]
    chart: LocalChart[Any]

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Array:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        duration = self.t1 - self.t0
        fraction = jnp.asarray(t0 - self.t0) / jnp.where(duration == 0, 1, duration)
        y = self.y0
        for increment in self.increments:
            y = self.geometry.apply_increment(y, fraction * increment, self.chart)
        # Preserve the exact stored endpoints, including zero-duration steps.
        y = jnp.where(t0 == self.t1, self.y1, y)
        return jnp.where(t0 == self.t0, self.y0, y)


def geometric_dense_info(
    y0: Array,
    y1: Array,
    increments: Sequence[Array],
    geometry: Manifold[Any],
    chart: LocalChart[Any],
) -> DenseInfo:
    return dict(
        y0=y0, y1=y1, increments=tuple(increments), geometry=geometry, chart=chart
    )
