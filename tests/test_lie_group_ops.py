from __future__ import annotations

import jax.numpy as jnp
import pytest
from conftest import EuclideanOps
from diffrax import ODETerm

from georax import SO, GeometricTerm, LieGroup


def _zero_vf(t, y, args):
    del t, y, args
    return 0.0


def test_chart_differential_inv_is_identity_at_zero_for_so3() -> None:
    term = GeometricTerm(inner=ODETerm(_zero_vf), geometry=SO(3))
    omega = jnp.zeros(SO(3).lie_algebra_dimension)
    eta = jnp.array([0.3, -0.4, 0.2])

    assert bool(jnp.allclose(term.chart_differential_inv(omega, eta), eta))


def test_non_lie_group_geometry_rejects_lie_group_helpers() -> None:
    term = GeometricTerm(inner=ODETerm(_zero_vf), geometry=EuclideanOps())
    a = jnp.array([0.1, -0.2])

    with pytest.raises(TypeError, match="LieGroupOps"):
        term.chart_differential_inv(a, a)
