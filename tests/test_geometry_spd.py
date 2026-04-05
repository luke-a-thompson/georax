from __future__ import annotations

import jax.numpy as jnp
from diffrax import ODETerm

from georax import CG2, SPD, GeometricTerm


def _is_spd(x: jnp.ndarray) -> bool:
    return bool(
        jnp.allclose(x, x.T, atol=1e-6) and jnp.all(jnp.linalg.eigvalsh(x) > 0.0)
    )


def test_spd_to_frame_from_frame_roundtrip() -> None:
    spd = SPD(3)
    x = jnp.array([[1.7, 0.2, 0.1], [0.2, 1.3, 0.05], [0.1, 0.05, 0.9]])
    a = jnp.array([1.2, 0.8, 1.5, -0.4, 0.3, 0.6])

    v = spd.from_frame(x, a)
    recovered = spd.to_frame(x, v)

    assert bool(jnp.allclose(v, v.T))
    assert bool(jnp.allclose(recovered, a))


def test_spd_retraction_stays_in_spd() -> None:
    spd = SPD(2)
    x = jnp.array([[2.0, 0.3], [0.3, 1.4]])
    a = jnp.array([0.2, -0.1, 0.35])

    v = spd.from_frame(x, a)
    y = spd.retraction(x, v)

    assert _is_spd(y)


def test_spd_retraction_matches_first_order_tangent_step() -> None:
    spd = SPD(2)
    x = jnp.array([[1.8, 0.2], [0.2, 1.3]])
    a = jnp.array([0.4, -0.15, 0.25])
    v = spd.from_frame(x, a)
    eps = 1e-4

    y = spd.retraction(x, eps * v)

    assert bool(jnp.allclose(y, x + eps * v, atol=1e-7, rtol=1e-4))


def test_spd_commutator_free_step_preserves_spd() -> None:
    geometry = SPD(2)
    solver = CG2()
    y0 = jnp.array([[1.5, 0.1], [0.1, 1.2]])

    def vf(t, y, args):
        del t, y, args
        return jnp.array([[0.15, 0.05], [0.05, -0.02]])

    term = GeometricTerm(inner=ODETerm(vf), geometry=geometry)
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
