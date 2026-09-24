from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._charts import (
    SPDChart,
    _sym,
)
from .base import FrameCoords, Manifold, StateArray

__all__ = ["SPD"]


def _solve_symmetric_sylvester(x: Array, v: Array) -> Array:
    """Solve x a + a x = v, with derivatives defined by the equation."""
    x = _sym(x)

    def solve(_: Callable[[Array], Array], rhs: Array) -> Array:
        eigvals, eigvecs = jnp.linalg.eigh(x)
        local_rhs = eigvecs.T @ rhs @ eigvecs
        local_a = local_rhs / (eigvals[:, None] + eigvals[None, :])
        return eigvecs @ local_a @ eigvecs.T

    # Implicit differentiation avoids eigenvector derivatives, including for
    # higher derivatives at repeated eigenvalues such as the identity.
    return jax.lax.custom_linear_solve(
        lambda a: x @ a + a @ x, _sym(v), solve, symmetric=True
    )


class SPD(Manifold["SPD"]):
    """SPD(n) with scaled-``vech`` symmetric-lift coordinates.

    Coefficients are represented in the symmetric matrix algebra. The induced
    infinitesimal action is

        E_A(x) = A x + x A,

    and the local chart approximates the congruence action to the order selected by
    the solver:

        Phi_A(x) = exp(A) x exp(A).
    """

    _chart_class = SPDChart
    n: int = eqx.field(static=True)

    def __init__(self, n: int) -> None:
        n = int(n)
        if n < 1:
            raise ValueError("SPD(n) requires n >= 1.")

        object.__setattr__(self, "n", n)

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return (self.n * (self.n + 1) // 2,)

    def _coords_to_sym(self, a: FrameCoords) -> Array:
        self.check_coordinate_shape(a)
        coeffs = jnp.asarray(a)
        tangent = jnp.zeros((self.n, self.n), dtype=coeffs.dtype)
        diag_i = np.arange(self.n)
        tangent = tangent.at[diag_i, diag_i].set(coeffs[: self.n])
        if self.coordinate_shape[0] > self.n:
            sqrt_two = jnp.asarray(np.sqrt(2.0), dtype=coeffs.dtype)
            off_diag = coeffs[self.n :] / sqrt_two
            upper_i, upper_j = np.triu_indices(self.n, k=1)
            tangent = tangent.at[upper_i, upper_j].set(off_diag)
            tangent = tangent.at[upper_j, upper_i].set(off_diag)
        return tangent

    def _sym_to_coords(self, tangent: Array) -> FrameCoords:
        if tangent.shape != self.state_shape:
            raise ValueError(
                f"{type(self).__name__} symmetric matrix must have shape {self.state_shape}; got {tangent.shape}."
            )
        tangent = _sym(jnp.asarray(tangent))
        diag = jnp.diag(tangent)
        sqrt_two = jnp.asarray(np.sqrt(2.0), dtype=tangent.dtype)
        off_diag = sqrt_two * tangent[np.triu_indices(self.n, k=1)]
        return jnp.concatenate((diag, off_diag))

    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        self.check_state_shape(x)
        self.check_state_shape(v)
        return self._sym_to_coords(_solve_symmetric_sylvester(x, v))

    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        self.check_state_shape(x)
        lift = self._coords_to_sym(a)
        return lift @ x + x @ lift

    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        self.check_state_shape(x)
        lift_a = self._coords_to_sym(a)
        lift_b = self._coords_to_sym(b)
        skew = lift_a @ lift_b - lift_b @ lift_a
        return self.trivialise(x, x @ skew - skew @ x)
