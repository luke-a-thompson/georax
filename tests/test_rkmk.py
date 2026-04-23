from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import pytest
from conftest import make_solver_accuracy_term
from diffrax import Heun

from georax import CG2, RKMK, Euclidean, GeometricTerm

jax.config.update("jax_enable_x64", True)

_T0 = 0.0
_T1 = 1.0
_Y0 = jnp.eye(3, dtype=jnp.float64)
_MAX_STEPS = 100_000
_TERM = make_solver_accuracy_term()
_REFERENCE = diffrax.diffeqsolve(
    _TERM.inner,
    diffrax.Dopri8(),
    _T0,
    _T1,
    1e-3,
    _Y0,
    saveat=diffrax.SaveAt(t1=True),
    stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12),
    max_steps=_MAX_STEPS,
    throw=True,
)
assert _REFERENCE.ys is not None
_REFERENCE_Y1 = _REFERENCE.ys[0]


def test_rkmk_rejects_non_erk_base_solver() -> None:
    with pytest.raises(TypeError):
        RKMK(CG2())


def test_rkmk_rejects_non_ode_inner_term() -> None:
    control_term = GeometricTerm(
        inner=diffrax.ControlTerm(
            lambda t, y, args: jnp.ones((1, 1)),
            lambda t0, t1: jnp.array([t1 - t0]),
        ),
        geometry=Euclidean(),
    )
    solver = RKMK(Heun())

    with pytest.raises(TypeError, match="ODETerm"):
        solver.step(
            terms=control_term,
            t0=0.0,
            t1=1.0,
            y0=jnp.array([1.0]),
            args=None,
            solver_state=None,
            made_jump=False,
        )


def test_rkmk_heun_preserves_so3_orthogonality() -> None:
    solver = RKMK(Heun())
    out = diffrax.diffeqsolve(
        _TERM,
        solver,
        _T0,
        _T1,
        0.01,
        _Y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=_MAX_STEPS,
        throw=True,
    )
    assert out.ys is not None
    R = out.ys[0]

    assert bool(jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-10))
    assert float(jnp.linalg.det(R)) > 0.0


def test_rkmk_heun_empirical_order_matches_base() -> None:
    solver = RKMK(Heun())
    expected_order = solver.order(_TERM)
    assert expected_order == 2

    dts = jnp.array([0.1, 0.05, 0.025, 0.0125], dtype=jnp.float64)
    errors = []
    for dt in dts:
        out = diffrax.diffeqsolve(
            _TERM,
            solver,
            _T0,
            _T1,
            float(dt),
            _Y0,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=_MAX_STEPS,
            throw=True,
        )
        assert out.ys is not None
        errors.append(float(jnp.linalg.norm(out.ys[0] - _REFERENCE_Y1)))

    log_dts = jnp.log(dts)
    log_errs = jnp.log(jnp.array(errors, dtype=jnp.float64))
    slope = float(jnp.polyfit(log_dts, log_errs, 1)[0])

    assert slope >= expected_order * 0.9, (
        f"Expected order ~{expected_order}, got slope {slope:.2f}"
    )


def test_rkmk_heun_returns_error_estimate_from_embedded_base() -> None:
    solver = RKMK(Heun())

    _, y_error, _, _, result = solver.step(
        terms=_TERM,
        t0=_T0,
        t1=0.1,
        y0=_Y0,
        args=None,
        solver_state=None,
        made_jump=False,
    )
    assert result == diffrax.RESULTS.successful
    assert y_error is not None
    assert float(jnp.linalg.norm(y_error)) > 0.0
