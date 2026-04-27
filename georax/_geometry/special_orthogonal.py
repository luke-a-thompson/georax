from __future__ import annotations

from typing import override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from diffrax._custom_types import RealScalarLike
from jaxtyping import Array

from .base import LocalChart, Manifold
from ._charts import CayleyChart, ExpChart, PadeChart, RodriguesChart

__all__ = ["SO"]


class SO(Manifold):
    """SO(n) with a left-invariant frame and Cayley retraction."""

    n: int = eqx.field(static=True)
    _upper_i: Array
    _upper_j: Array
    _basis: Array

    def __init__(self, n: int, *, chart: LocalChart | None = None):
        n = int(n)
        if n < 2:
            raise ValueError("SO(n) requires n >= 2.")

        upper_i, upper_j = np.triu_indices(n, k=1)
        d = upper_i.size
        basis = np.zeros((n, n, d), dtype=float)
        k = np.arange(d)
        basis[upper_i, upper_j, k] = 1.0
        basis[upper_j, upper_i, k] = -1.0

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "_upper_i", jnp.asarray(upper_i))
        object.__setattr__(self, "_upper_j", jnp.asarray(upper_j))
        object.__setattr__(self, "_basis", jnp.asarray(basis))
        object.__setattr__(self, "chart", CayleyChart() if chart is None else chart)

    @property
    def lie_algebra_dimension(self) -> int:
        return int(self._upper_i.size)

    def _coords_to_alg(self, a: Array, *, dtype=None) -> Array:
        coeffs = jnp.asarray(a, dtype=dtype)
        omega = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        omega = omega.at[self._upper_i, self._upper_j].set(coeffs)
        omega = omega.at[self._upper_j, self._upper_i].set(-coeffs)
        return omega

    def _alg_to_coords(self, omega: Array) -> Array:
        omega = 0.5 * (omega - omega.T)
        return omega[self._upper_i, self._upper_j]

    @override
    def select_chart(self, required_order: RealScalarLike) -> LocalChart:
        if required_order == "exact":
            chart: LocalChart = RodriguesChart() if self.n == 3 else ExpChart()
        elif required_order <= 2:
            chart = CayleyChart()
        elif self.n == 3:
            chart = RodriguesChart()
        else:
            chart = PadeChart(int(required_order))

        object.__setattr__(self, "chart", chart)
        return chart
