from __future__ import annotations

import jax
import jax.numpy as jnp

from georax import CG2, SPD, GeometricTerm, post_lie_bracket


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
