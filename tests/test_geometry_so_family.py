from __future__ import annotations

import jax
import jax.numpy as jnp

from georax import SO


def test_so_increment_stays_on_group() -> None:
    so5 = SO(5)
    x = jnp.eye(5)
    a = jnp.linspace(-0.2, 0.2, so5.lie_algebra_dimension)

    y = so5.apply_increment(x, a)

    assert bool(jnp.allclose(y.T @ y, jnp.eye(5), atol=1e-6))
    assert float(jnp.linalg.det(y)) > 0.0


def test_chart_inverse_differential_is_identity_at_zero() -> None:
    so3 = SO(3)
    so3.select_chart(2)
    omega = jnp.zeros(so3.lie_algebra_dimension)
    eta = jnp.array([0.3, -0.4, 0.2])
    assert so3.chart is not None

    corrected = so3.chart.inverse_differential(jnp.eye(3), omega, eta, so3)
    assert bool(jnp.allclose(corrected, eta))


def test_cayley_inverse_differential_matches_chart_jvp() -> None:
    so3 = SO(3)
    chart = so3.select_chart(2)
    x = jnp.eye(3)
    omega = jnp.array([0.2, -0.1, 0.15])
    eta = jnp.array([0.3, -0.4, 0.2])

    pullback = chart.inverse_differential(x, omega, eta, so3)
    y, tangent = jax.jvp(
        lambda a: chart.apply(x, a, so3),
        (omega,),
        (pullback,),
    )
    target = y @ so3._coords_to_alg(eta, dtype=y.dtype)

    assert bool(jnp.allclose(tangent, target, atol=1e-6))
