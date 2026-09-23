from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from georax import CG2, RKMK, SPD, GeometricTerm
from georax._geometry.base import post_lie_bracket

jax.config.update("jax_enable_x64", True)


def _is_spd(x: jnp.ndarray) -> bool:
    return bool(
        jnp.allclose(x, x.T, atol=1e-6) and jnp.all(jnp.linalg.eigvalsh(x) > 0.0)
    )


def test_spd_increment_stays_in_spd() -> None:
    spd = SPD(2)
    spd.select_chart(12)
    x = jnp.array([[2.0, 0.3], [0.3, 1.4]])
    a = jnp.array([0.2, -0.1, 0.35])

    y = spd.apply_increment(x, a)

    assert _is_spd(y)


def test_spd_increment_matches_first_order_tangent_step() -> None:
    spd = SPD(2)
    spd.select_chart(12)
    x = jnp.array([[1.8, 0.2], [0.2, 1.3]])
    a = jnp.array([0.4, -0.15, 0.25])
    lift = spd._coords_to_sym(a)
    v = lift @ x + x @ lift
    eps = 1e-4

    y = spd.apply_increment(x, eps * a)

    assert bool(jnp.allclose(y, x + eps * v, atol=1e-7, rtol=1e-4))


def test_spd_frame_bracket_matches_ambient_lie_bracket() -> None:
    spd = SPD(2)
    x = jnp.array([[1.8, 0.25], [0.25, 1.2]])
    a = jnp.array([0.4, -0.15, 0.25])
    b = jnp.array([-0.2, 0.35, 0.1])

    def vector_a(y):
        return spd.detrivialise(y, a)

    def vector_b(y):
        return spd.detrivialise(y, b)

    ambient_bracket = (
        jax.jvp(vector_b, (x,), (vector_a(x),))[1]
        - jax.jvp(vector_a, (x,), (vector_b(x),))[1]
    )
    expected = spd.trivialise(x, ambient_bracket)

    bracket = spd.frame_bracket(x, a, b)

    assert bool(jnp.allclose(bracket, expected, atol=1e-6))


def test_spd_post_lie_bracket_supports_constant_frame_fields() -> None:
    spd = SPD(2)
    x = jnp.array([[1.8, 0.25], [0.25, 1.2]])
    a = jnp.array([0.4, -0.15, 0.25])
    b = jnp.array([-0.2, 0.35, 0.1])

    bracket = post_lie_bracket(spd, lambda y: a, lambda y: b, x)

    assert bool(jnp.allclose(bracket, spd.frame_bracket(x, a, b), atol=1e-6))


def test_spd_commutator_free_step_preserves_spd() -> None:
    geometry = SPD(2)
    solver = CG2()
    y0 = jnp.array([[1.5, 0.1], [0.1, 1.2]])

    def coeffs(t, y, args):
        del t, y, args
        return jnp.array([0.05, -0.01, 0.02])

    term = GeometricTerm(coeffs, geometry=geometry)
    solver_state = solver.init(term, 0.0, 0.1, y0, None)
    y1, _, _, _, _ = solver.step(
        terms=term,
        t0=0.0,
        t1=0.1,
        y0=y0,
        args=None,
        solver_state=solver_state,
        made_jump=False,
    )

    assert _is_spd(y1)


@pytest.mark.parametrize("diagonal", [(1.0, 1.0, 1.0), (1.0, 1.0, 2.0)])
def test_spd_trivialise_derivative_at_repeated_eigenvalues(diagonal):
    geometry = SPD(3)
    x = jnp.diag(jnp.array(diagonal))
    v = jnp.array([[0.3, 0.2, -0.1], [0.2, 0.1, 0.4], [-0.1, 0.4, 0.5]])
    dx = jnp.array([[0.1, -0.2, 0.0], [-0.2, 0.3, 0.1], [0.0, 0.1, -0.1]])
    dv = 0.2 * v
    a, da = jax.jvp(geometry.trivialise, (x, v), (dx, dv))
    lift = geometry._coords_to_sym(a)
    np.testing.assert_allclose(
        geometry.detrivialise(x, da) + dx @ lift + lift @ dx, dv, atol=1e-12
    )


def test_spd_solve_initial_state_gradient_at_identity():
    term = GeometricTerm(lambda t, y, args: jnp.array([0.1, 0.2, 0.05]), SPD(2))

    def loss(x):
        return diffrax.diffeqsolve(
            term, RKMK(diffrax.Heun()), t0=0.0, t1=0.1, dt0=0.1, y0=x
        ).ys.sum()

    x = jnp.eye(2)
    direction = jnp.array([[0.3, 0.2], [0.2, -0.1]])
    gradient = jax.jit(jax.grad(loss))(x)
    assert jnp.all(jnp.isfinite(gradient))
    eps = 1e-5
    expected = (loss(x + eps * direction) - loss(x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(jnp.sum(gradient * direction), expected, rtol=1e-7)


def test_spd_trivialise_second_derivative_at_identity():
    geometry = SPD(2)
    v = jnp.array([[0.3, 0.1], [0.1, 0.2]])
    loss = lambda scale: geometry.trivialise(scale * jnp.eye(2), v).sum()
    np.testing.assert_allclose(jax.hessian(loss)(1.0), 2 * loss(1.0), atol=1e-12)
