from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from georax import CFEES25, CFEES27, Euclidean, GeometricTerm, SO, SPD


# Resolve local truncation errors without single-precision cancellation.
jax.config.update("jax_enable_x64", True)


def _step(solver, term, y0, t0, t1):
    state = solver.init(term, t0, t1, y0, None)
    return solver.step(term, t0, t1, y0, None, state, False)


@pytest.mark.parametrize(
    "solver_cls, mu", [(CFEES25, 9 / 8), (CFEES27, 3 * np.sqrt(2) / 4)]
)
@pytest.mark.parametrize("h", [0.2, -0.2])
def test_embedded_companion_coefficients_and_evaluation_count(
    solver_cls, mu, h, monkeypatch
):
    solver = solver_cls()
    evaluations = []
    actions = []
    apply_increment = Euclidean.apply_increment

    def counted_action(self, x, a, chart):
        actions.append(a)
        return apply_increment(self, x, a, chart)

    monkeypatch.setattr(Euclidean, "apply_increment", counted_action)

    def vf(t, y, args):
        evaluations.append(t)
        return jnp.full_like(y, t)

    term = GeometricTerm(vf, geometry=Euclidean())
    y0 = jnp.array([2.0])
    t0 = 0.3
    y1, error, dense_info, state, result = _step(solver, term, y0, t0, t0 + h)

    # For y' = t, the principal method is exact and the companion has
    # y_hat = y0 + h*t0 + mu*h^2. These moments identify the intended pair.
    np.testing.assert_allclose(y1, y0 + h * t0 + h**2 / 2, atol=1e-14)
    np.testing.assert_allclose(y1 - error, y0 + h * t0 + mu * h**2, atol=1e-14)
    np.testing.assert_allclose(error, (0.5 - mu) * h**2, atol=1e-14)
    assert len(evaluations) == solver.recurrence.num_stages
    assert len(actions) == solver.recurrence.num_stages + 1
    assert len(dense_info["increments"]) == solver.recurrence.num_stages
    np.testing.assert_array_equal(state, y1)
    assert solver.error_order(term) == 2
    assert result == diffrax.RESULTS.successful


@pytest.mark.parametrize("solver_cls", [CFEES25, CFEES27])
def test_embedded_and_principal_local_orders(solver_cls):
    solver = solver_cls()
    term = GeometricTerm(lambda t, y, args: y, geometry=Euclidean())
    hs = np.array([0.1, 0.05, 0.025, 0.0125])
    estimates = []
    principal_errors = []
    companion_errors = []
    for h in hs:
        y1, error, _, _, _ = _step(solver, term, jnp.ones(1), 0.0, h)
        estimates.append(float(jnp.linalg.norm(error)))
        principal_errors.append(float(jnp.linalg.norm(y1 - np.exp(h))))
        companion_errors.append(float(jnp.linalg.norm(y1 - error - np.exp(h))))

    for errors, expected in [
        (estimates, 2), (principal_errors, 3), (companion_errors, 2)
    ]:
        slope = np.polyfit(np.log(hs), np.log(errors), 1)[0]
        assert abs(slope - expected) < 0.1


@pytest.mark.parametrize(
    "solver_cls, mu", [(CFEES25, 9 / 8), (CFEES27, 3 * np.sqrt(2) / 4)]
)
@pytest.mark.parametrize("geometry", [SO(2), SPD(1)])
def test_embedded_companion_is_anchored_at_initial_manifold_point(
    solver_cls, mu, geometry
):
    solver = solver_cls()
    h = 0.1
    term = GeometricTerm(lambda t, y, args: jnp.array([t]), geometry=geometry)
    if isinstance(geometry, SO):
        theta = 0.4
        y0 = jnp.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        angle = theta + mu * h**2
        expected = jnp.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
    else:
        y0 = jnp.array([[2.0]])
        expected = y0 * np.exp(2 * mu * h**2)

    y1, error, _, _, _ = _step(solver, term, y0, 0.0, h)
    y_hat = y1 - error
    np.testing.assert_allclose(y_hat, expected, rtol=1e-12, atol=1e-12)
    for y in (y1, y_hat):
        if isinstance(geometry, SO):
            np.testing.assert_allclose(y.T @ y, jnp.eye(2), atol=1e-12)
            np.testing.assert_allclose(jnp.linalg.det(y), 1.0, atol=1e-12)
        else:
            assert bool(jnp.all(jnp.linalg.eigvalsh(y) > 0))


@pytest.mark.parametrize("solver_cls", [CFEES25, CFEES27])
@pytest.mark.parametrize("direction", [1, -1])
def test_pid_rejects_restarts_and_retains_second_order_solution(solver_cls, direction):
    term = GeometricTerm(lambda t, y, args: y, geometry=Euclidean())
    t0, t1 = (0.0, 1.0) if direction == 1 else (1.0, 0.0)
    solution = diffrax.diffeqsolve(
        term,
        solver_cls(),
        t0=t0,
        t1=t1,
        dt0=direction * 0.5,
        y0=jnp.array([np.exp(t0)]),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=2048,
    )
    assert int(solution.stats["num_rejected_steps"]) > 0
    assert int(solution.stats["num_accepted_steps"]) > 1
    np.testing.assert_allclose(solution.ys[0], np.exp(t1), rtol=1e-5)


@pytest.mark.parametrize("solver_cls", [CFEES25, CFEES27])
def test_zero_discrepancy_allows_step_growth(solver_cls):
    term = GeometricTerm(lambda t, y, args: jnp.zeros_like(y), geometry=Euclidean())
    solution = diffrax.diffeqsolve(
        term,
        solver_cls(),
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=jnp.ones(1),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
        max_steps=10,
    )
    assert int(solution.stats["num_rejected_steps"]) == 0
    assert int(solution.stats["num_accepted_steps"]) < 10
    np.testing.assert_array_equal(solution.ys[0], jnp.ones(1))
