from __future__ import annotations

import jax.numpy as jnp
from diffrax import ControlTerm

from georax import CFEES25, GeometricTerm, SPDHomogeneous


def _is_spd(x: jnp.ndarray) -> bool:
    return bool(
        jnp.allclose(x, x.T, atol=1e-6) and jnp.all(jnp.linalg.eigvalsh(x) > 0.0)
    )


def test_spd_homogeneous_to_frame_from_frame_roundtrip() -> None:
    geometry = SPDHomogeneous(2)
    x = jnp.array([[1.7, 0.2], [0.2, 1.1]])
    alg = jnp.array([[0.3, -0.1], [0.2, 0.4]])

    v = geometry.from_frame(x, alg)
    recovered = geometry.from_frame(x, geometry.to_frame(x, v))

    assert bool(jnp.allclose(recovered, v))


def test_spd_homogeneous_frozen_flow_stays_in_spd() -> None:
    geometry = SPDHomogeneous(2)
    x = jnp.array([[1.8, 0.15], [0.15, 1.2]])
    alg = jnp.array([[0.2, -0.15], [0.1, 0.05]])

    y = geometry.frozen_flow(x, alg)

    assert _is_spd(y)


def test_spd_homogeneous_control_term_step_preserves_spd() -> None:
    geometry = SPDHomogeneous(2)
    control = jnp.array([0.04, -0.03])
    term = GeometricTerm(
        inner=ControlTerm(
            lambda t, x, args: geometry.frame(x)[..., :2],
            lambda t0, t1: control,
        ),
        geometry=geometry,
    )
    y0 = jnp.array([[1.5, 0.1], [0.1, 1.2]])

    y1, _, _, _, _ = CFEES25().step(
        terms=term,
        t0=0.0,
        t1=1.0,
        y0=y0,
        args=None,
        solver_state=None,
        made_jump=False,
    )

    assert _is_spd(y1)


def test_spd_homogeneous_coeffs_prod_override_recovers_exact_algebra_increment() -> None:
    geometry = SPDHomogeneous(2)
    control = jnp.array([0.04, -0.03])
    y0 = jnp.array([[1.5, 0.1], [0.1, 1.2]])
    expected_coeffs = jnp.array([0.04, -0.03, 0.0, 0.0])

    term = GeometricTerm(
        inner=ControlTerm(
            lambda t, x, args: geometry.frame(x)[..., :2],
            lambda t0, t1: control,
        ),
        geometry=geometry,
        coeffs_prod_fn=lambda t, x, args, control: expected_coeffs,
    )

    recovered = term.coeffs_prod(0.0, y0, None, control)
    y1, _, _, _, _ = CFEES25().step(
        terms=term,
        t0=0.0,
        t1=1.0,
        y0=y0,
        args=None,
        solver_state=None,
        made_jump=False,
    )

    assert bool(jnp.allclose(recovered, expected_coeffs))
    assert bool(jnp.allclose(y1, geometry.frozen_flow(y0, expected_coeffs)))
