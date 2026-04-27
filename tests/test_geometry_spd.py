from __future__ import annotations

import jax.numpy as jnp

from georax import CG2, SPD, GeometricTerm


def _is_spd(x: jnp.ndarray) -> bool:
    return bool(
        jnp.allclose(x, x.T, atol=1e-6) and jnp.all(jnp.linalg.eigvalsh(x) > 0.0)
    )


def test_spd_increment_stays_in_spd() -> None:
    spd = SPD(2)
    x = jnp.array([[2.0, 0.3], [0.3, 1.4]])
    a = jnp.array([0.2, -0.1, 0.35])

    y = spd.apply_increment(x, a)

    assert _is_spd(y)


def test_spd_increment_matches_first_order_tangent_step() -> None:
    spd = SPD(2)
    x = jnp.array([[1.8, 0.2], [0.2, 1.3]])
    a = jnp.array([0.4, -0.15, 0.25])
    lift = spd._coords_to_sym(a)
    v = lift @ x + x @ lift
    eps = 1e-4

    y = spd.apply_increment(x, eps * a)

    assert bool(jnp.allclose(y, x + eps * v, atol=1e-7, rtol=1e-4))


def test_spd_commutator_free_step_preserves_spd() -> None:
    geometry = SPD(2)
    solver = CG2()
    y0 = jnp.array([[1.5, 0.1], [0.1, 1.2]])

    def coeffs(t, y, args):
        del t, y, args
        return jnp.array([0.05, -0.01, 0.02])

    term = GeometricTerm(coeffs, geometry=geometry)
    y1, _, _, _, _ = solver.step(
        terms=term,
        t0=0.0,
        t1=0.1,
        y0=y0,
        args=None,
        solver_state=None,
        made_jump=False,
    )

    assert _is_spd(y1)
