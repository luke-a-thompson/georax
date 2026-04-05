from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import pytest
from conftest import SOLVERS, make_solver_accuracy_term

# Improves stability of empirical slope estimation for convergence-order checks.
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


def _reversible_roundtrip_error(
    solver: diffrax.AbstractReversibleSolver, dt: float
) -> float:
    num_steps = round((_T1 - _T0) / dt)
    assert abs(num_steps * dt - (_T1 - _T0)) < 1e-12

    ts = [_T0 + i * dt for i in range(num_steps + 1)]
    y = _Y0
    solver_state = solver.init(_TERM, ts[0], ts[1], y, None)

    for i in range(num_steps):
        y, _, _, solver_state, result = solver.step(
            _TERM, ts[i], ts[i + 1], y, None, solver_state, False
        )
        assert result == diffrax.RESULTS.successful

    for i in range(num_steps, 0, -1):
        tm1 = ts[i - 2] if i - 2 >= 0 else ts[i - 1]
        y, _, solver_state, result = solver.backward_step(
            _TERM, ts[i - 1], ts[i], y, None, (tm1,), solver_state, False
        )
        assert result == diffrax.RESULTS.successful

    return float(jnp.linalg.norm(y - _Y0))


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_solver_fixed_step_matches_reference_solution(solver_name, solver_cls):
    del solver_name

    solver = solver_cls()
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

    assert out.result == diffrax.RESULTS.successful, "Result reported as unsuccessful"
    assert out.ys is not None
    assert float(jnp.linalg.norm(out.ys[0] - _REFERENCE_Y1)) < 1e-3


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_solver_empirical_order_matches_declared_order(solver_name, solver_cls):
    del solver_name

    solver = solver_cls()
    expected_order = solver.order(_TERM)
    assert expected_order is not None

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


@pytest.mark.parametrize(("solver_name", "solver_cls"), SOLVERS)
def test_solver_backward_empirical_order_matches_declared_order(
    solver_name, solver_cls
):
    del solver_name

    solver = solver_cls()
    if not isinstance(solver, diffrax.AbstractReversibleSolver):
        pytest.skip("Backward-order checks apply only to reversible solvers.")
    expected_order = solver.antisymmetric_order(_TERM)
    assert expected_order is not None

    dts = jnp.array([0.1, 0.05, 0.025, 0.0125], dtype=jnp.float64)
    errors = []
    for dt in dts:
        errors.append(_reversible_roundtrip_error(solver, float(dt)))

    log_dts = jnp.log(dts)
    log_errs = jnp.log(jnp.array(errors, dtype=jnp.float64))
    slope = float(jnp.polyfit(log_dts, log_errs, 1)[0])

    assert slope >= expected_order * 0.9, (
        f"Expected backward order ~{expected_order}, got slope {slope:.2f}"
    )
