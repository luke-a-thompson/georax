from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
from conftest import make_solver_accuracy_ambient_term, make_solver_accuracy_term

from georax import CG4, RKMK, SO

jax.config.update("jax_enable_x64", True)

_T0 = 0.0
_T1 = 1.0
_Y0 = jnp.eye(3, dtype=jnp.float64)
_MAX_STEPS = 100_000


def test_high_order_so_inverse_differential_fallback_inverts_linearized_chart() -> None:
    geometry = SO(3)
    chart = geometry.select_chart(5)
    x = jnp.eye(3, dtype=jnp.float64)
    a = jnp.array([0.04, -0.03, 0.02], dtype=jnp.float64)
    b = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float64)

    delta = chart.inverse_differential(x, a, b, geometry)
    y, pushforward = jax.linearize(lambda aa: chart.apply(x, aa, geometry), a)
    reconstructed = geometry.trivialise(y, pushforward(delta))

    assert jnp.allclose(reconstructed, b, rtol=1e-10, atol=1e-10)


def test_rkmk_tsit5_converges_with_cayley_pullback() -> None:
    term = make_solver_accuracy_term()
    solver = RKMK(diffrax.Tsit5())
    expected_order = solver.order(term)
    assert expected_order == 5

    solver.init(term, _T0, _T1, _Y0, None)
    assert term.geometry.chart is not None
    assert term.geometry.chart.order == 2

    reference = diffrax.diffeqsolve(
        make_solver_accuracy_ambient_term(),
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
    assert reference.ys is not None
    reference_y1 = reference.ys[0]

    dts = jnp.array([0.4, 0.2, 0.1, 0.05], dtype=jnp.float64)
    errors = []
    for dt in dts:
        out = diffrax.diffeqsolve(
            term,
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
        errors.append(float(jnp.linalg.norm(out.ys[0] - reference_y1)))

    log_dts = jnp.log(dts)
    log_errs = jnp.log(jnp.array(errors, dtype=jnp.float64))
    slope = float(jnp.polyfit(log_dts, log_errs, 1)[0])

    assert slope >= expected_order * 0.9, (
        f"Expected order ~{expected_order}, got slope {slope:.2f}"
    )


def test_commutator_free_solver_still_selects_order_matched_chart() -> None:
    term = make_solver_accuracy_term()
    CG4().init(term, _T0, _T1, _Y0, None)

    assert term.geometry.chart is not None
    assert term.geometry.chart.order == 4
