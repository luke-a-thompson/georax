from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array


class GeometricOps(eqx.Module):
    """Minimal geometry backend for manifold Crouch--Grossman with retractions.

    This class contains only geometric primitives.
    It does not contain solver logic.
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
