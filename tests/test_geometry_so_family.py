from __future__ import annotations

import jax.numpy as jnp

from georax import SO


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


def test_chart_inverse_differential_is_identity_at_zero() -> None:
    so3 = SO(3)
    omega = jnp.zeros(so3.lie_algebra_dimension)
    eta = jnp.array([0.3, -0.4, 0.2])
    assert so3.chart is not None

    corrected = so3.chart.inverse_differential(jnp.eye(3), omega, eta, so3)
    assert bool(jnp.allclose(corrected, eta))
