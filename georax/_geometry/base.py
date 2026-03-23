from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array


class Manifold(eqx.Module):
    """Minimal manifold geometry backend for retraction-based integrators.

    This class contains only geometric primitives and does not contain solver
    logic. It is sufficient for retraction-based manifold Runge--Kutta schemes
    such as the current commutator-free solvers.
    """

    @abstractmethod
    def frame(self, x: Array) -> Array:
        """Return a local frame at x.

        Convention:
            shape = ambient_shape + (d,)
        where d = manifold dimension, and the last axis indexes frame vectors.
        """
        ...

    @abstractmethod
    def to_frame(self, x: Array, v: Array) -> Array:
        """Return coefficients a such that v = sum_i a_i E_i(x)."""
        ...

    @abstractmethod
    def from_frame(self, x: Array, a: Array) -> Array:
        """Return the tangent vector sum_i a_i E_i(x)."""
        ...

    @abstractmethod
    def retraction(self, x: Array, v: Array) -> Array:
        """Map a tangent vector v in T_xM back to M."""
        ...


class LieGroup(Manifold):
    """Lie-group geometry with a chosen chart for Lie-algebra integrators.

    The chart is understood as a local map tau from Lie-algebra coordinates to
    the group near the identity. For RKMK-style methods one needs the inverse
    of the left-trivialized chart differential at a.
    """

    @abstractmethod
    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        """Apply the inverse left-trivialized chart differential at a to b."""
        ...
