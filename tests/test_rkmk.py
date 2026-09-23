from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
from _problems import make_solver_accuracy_ambient_term, make_solver_accuracy_term
from diffrax import Heun

from georax import RKMK, Euclidean, GeometricTerm
from georax._term import PulledDriftTerm

jax.config.update("jax_enable_x64", True)

_T0 = 0.0
_T1 = 1.0
_Y0 = jnp.eye(3, dtype=jnp.float64)
_MAX_STEPS = 100_000
_TERM = make_solver_accuracy_term()


def _reference_y1():
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
    return reference.ys[0]


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
    reference_y1 = _reference_y1()

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
        errors.append(float(jnp.linalg.norm(out.ys[0] - reference_y1)))

    log_dts = jnp.log(dts)
    log_errs = jnp.log(jnp.array(errors, dtype=jnp.float64))
    slope = float(jnp.polyfit(log_dts, log_errs, 1)[0])

    assert slope >= expected_order * 0.9, (
        f"Expected order ~{expected_order}, got slope {slope:.2f}"
    )


def test_rkmk_heun_returns_error_estimate_from_embedded_base() -> None:
    solver = RKMK(Heun())
    out = diffrax.diffeqsolve(
        _TERM,
        solver,
        _T0,
        _T1,
        0.1,
        _Y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
        max_steps=_MAX_STEPS,
        throw=True,
    )
    assert out.ys is not None
    assert out.result == diffrax.RESULTS.successful
    assert out.stats["num_accepted_steps"] > 0


def test_rkmk_matches_heun_on_euclidean_geometry() -> None:
    def vf(t, y, args):
        del t, args
        return -0.2 * y

    y0 = jnp.array([1.0, -0.5])
    kwargs = dict(t0=0.0, t1=1.0, dt0=0.1, y0=y0, saveat=diffrax.SaveAt(t1=True))
    wrapped = diffrax.diffeqsolve(
        GeometricTerm(vf, Euclidean()), RKMK(Heun()), **kwargs
    )
    reference = diffrax.diffeqsolve(diffrax.ODETerm(vf), Heun(), **kwargs)

    assert wrapped.ys is not None
    assert reference.ys is not None
    assert bool(jnp.allclose(wrapped.ys, reference.ys, atol=1e-7))


def test_rkmk_error_is_difference_between_reconstructed_states() -> None:
    solver = RKMK(Heun())
    solver.init(_TERM, 0.0, 0.2, _Y0, None)
    y1, y_error, _, _, _ = solver.step(_TERM, 0.0, 0.2, _Y0, None, None, False)

    geometry = _TERM.geometry
    omega0 = geometry.zero_coordinates(_Y0)
    omega1, omega_error, _, _, _ = solver.solver.step(
        PulledDriftTerm(_TERM, _Y0),
        0.0,
        0.2,
        omega0,
        None,
        None,
        False,
    )
    assert omega_error is not None
    expected = geometry.apply_increment(_Y0, omega1 + omega_error) - y1

    assert y_error is not None
    assert bool(jnp.allclose(y_error, expected))
