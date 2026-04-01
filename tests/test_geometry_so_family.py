from __future__ import annotations

import jax.numpy as jnp
import pytest
from conftest import EuclideanOps
from diffrax import ODETerm

from georax import SO, GeometricTerm


def _zero_vf(t, y, args):
    del t, y, args
    return 0.0


def test_so_ops_to_frame_from_frame_roundtrip() -> None:
    so4 = SO(4)
    x = jnp.eye(4)
    a = jnp.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.6])

    v = so4.from_frame(x, a)
    recovered = so4.to_frame(x, v)

    assert bool(jnp.allclose(recovered, a))


def test_so_retraction_stays_on_group() -> None:
    so5 = SO(5)
    x = jnp.eye(5)
    a = jnp.linspace(-0.2, 0.2, so5.lie_algebra_dimension)

    v = so5.from_frame(x, a)
    y = so5.retraction(x, v)

    assert bool(jnp.allclose(y.T @ y, jnp.eye(5), atol=1e-6))
    assert float(jnp.linalg.det(y)) > 0.0


def test_chart_differential_inv_is_identity_at_zero() -> None:
    term = GeometricTerm(inner=ODETerm(_zero_vf), geometry=SO(3))
    omega = jnp.zeros(SO(3).lie_algebra_dimension)
    eta = jnp.array([0.3, -0.4, 0.2])

    assert bool(jnp.allclose(term.chart_differential_inv(omega, eta), eta))


def test_manifold_raises_on_lie_group_helpers() -> None:
    # EuclideanOps is a Manifold but not a LieGroup subclass; chart_differential_inv
    # requires LieGroup and should raise.
    term = GeometricTerm(inner=ODETerm(_zero_vf), geometry=EuclideanOps())
    a = jnp.array([0.1, -0.2])

    with pytest.raises(TypeError):
        term.chart_differential_inv(a, a)
