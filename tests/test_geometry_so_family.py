from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from georax import SO


def test_so_increment_stays_on_group() -> None:
    so5 = SO(5)
    chart = so5.select_chart(12)
    x = jnp.eye(5)
    a = jnp.linspace(-0.2, 0.2, so5.coordinate_shape[0])

    y = so5.apply_increment(x, a, chart)

    assert bool(jnp.allclose(y.T @ y, jnp.eye(5), atol=1e-6))
    assert float(jnp.linalg.det(y)) > 0.0


@pytest.mark.parametrize("n", [3, 5])
def test_so_frame_bracket_matches_matrix_commutator(n: int) -> None:
    so = SO(n)
    d = so.coordinate_shape[0]
    x = jnp.eye(n)
    a = jnp.linspace(-0.2, 0.3, d)
    b = jnp.linspace(0.4, -0.1, d)

    omega_a = so._coords_to_alg(a)
    omega_b = so._coords_to_alg(b)
    expected = so._alg_to_coords(omega_a @ omega_b - omega_b @ omega_a)

    assert bool(jnp.allclose(so.frame_bracket(x, a, b), expected, atol=1e-6))


def test_so_rejects_wrong_state_shape() -> None:
    so3 = SO(3)
    chart = so3.select_chart(2)

    with pytest.raises(ValueError, match=r"SO state must have shape \(3, 3\)"):
        so3.apply_increment(jnp.ones(4), jnp.zeros(3), chart)


def test_so_rejects_wrong_coordinate_shape() -> None:
    so3 = SO(3)
    chart = so3.select_chart(2)

    with pytest.raises(ValueError, match=r"SO coordinates must have shape \(3,\)"):
        so3.apply_increment(jnp.eye(3), jnp.zeros((3, 3)), chart)


def test_chart_inverse_differential_is_identity_at_zero() -> None:
    so3 = SO(3)
    chart = so3.select_chart(2)
    omega = jnp.zeros(so3.coordinate_shape)
    eta = jnp.array([0.3, -0.4, 0.2])

    corrected = chart.inverse_differential(jnp.eye(3), omega, eta, so3)
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
    target = y @ so3._coords_to_alg(eta)

    assert bool(jnp.allclose(tangent, target, atol=1e-6))
