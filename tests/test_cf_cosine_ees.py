from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax_lowstorage import CosineEES

from georax import CFCosineEES, Euclidean, GeometricTerm

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("order", [-3, 0, 1, 2, 4, 10, 3.0, 3.5, True, "9", None])
def test_invalid_reversible_order_is_rejected(order):
    with pytest.raises(ValueError, match="odd integer >= 3"):
        CFCosineEES(reversible_order=order)


@pytest.mark.parametrize("order", [3, 5, 7, 9, 11, 31])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("direction", [1, -1])
def test_euclidean_step_matches_upstream_under_jit(order, stochastic, direction):
    solver = CFCosineEES(reversible_order=order)
    base_solver = CosineEES(reversible_order=order)
    vf = lambda t, y, args: -t * y**2
    term = GeometricTerm(vf, Euclidean())
    base_term = diffrax.ODETerm(vf)
    if stochastic:
        diffusion = diffrax.ControlTerm(
            lambda t, y, args: y[:, None],
            lambda t0, t1, **kwargs: jnp.array([0.3 * (t1 - t0)]),
        )
        term = diffrax.MultiTerm(term, diffusion)
        base_term = diffrax.MultiTerm(base_term, diffusion)
    t0, t1 = 0.3, 0.3 + direction * 0.2
    y0 = jnp.array([0.2, 0.5])

    @jax.jit
    def step(method, y):
        state = method.init(term, t0, t1, y, None)
        return method.step(term, t0, t1, y, None, state, False)

    y1, error, _, state, result = step(solver, y0)
    expected = base_solver.step(base_term, t0, t1, y0, None, None, False)[0]
    np.testing.assert_allclose(y1, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(state, y1)
    assert error is None
    assert result == diffrax.RESULTS.successful
    assert solver.error_order(term) is None
    assert solver.order(term) == 2
    assert solver.strong_order(term) == 0.5
    assert solver.antisymmetric_order(term) == order
    assert solver.recurrence.num_stages == (order + 1) // 2
    assert isinstance(solver, diffrax.AbstractStratonovichSolver)
    assert isinstance(solver, diffrax.AbstractReversibleSolver)

    recovered, dense_info, state, result = solver.backward_step(
        term, t0, t1, y1, None, (), state, False
    )
    expected_back = base_solver.backward_step(
        base_term, t0, t1, expected, None, (), None, False
    )[0]
    np.testing.assert_allclose(recovered, expected_back, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(state, recovered)
    assert result == diffrax.RESULTS.successful
    interpolator = solver.interpolation_cls(t0=t0, t1=t1, **dense_info)
    np.testing.assert_array_equal(interpolator.evaluate(t0), recovered)
    expected_forward = base_solver.step(
        base_term, t0, t1, expected_back, None, None, False
    )[0]
    np.testing.assert_allclose(
        interpolator.evaluate(t1), expected_forward, rtol=1e-13, atol=1e-14
    )
