from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from georax import Euclidean, GeometricEulerHeun, GeometricTerm


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("operator", [False, True])
def test_geometric_euler_heun_matches_diffrax(backward, operator):
    # State/time dependence and nonzero drift distinguish a noise-only,
    # same-time diffusion predictor from a full Euler or endpoint predictor.
    def drift(t, y, args):
        return args * y + t

    def diffusion(t, y, args):
        diagonal = 0.2 + 0.1 * y + 0.05 * t
        return lx.DiagonalLinearOperator(diagonal) if operator else jnp.diag(diagonal)

    path = diffrax.VirtualBrownianTree(0.0, 1.0, 1e-4, (2,), jax.random.key(7))
    geo_terms = diffrax.MultiTerm(
        GeometricTerm(drift, Euclidean()), diffrax.ControlTerm(diffusion, path)
    )
    ref_terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, path)
    )
    solver = GeometricEulerHeun()
    assert isinstance(solver, diffrax.AbstractStratonovichSolver)
    assert not isinstance(solver, diffrax.AbstractItoSolver)
    assert solver.order(geo_terms) == 1
    assert solver.strong_order(geo_terms) == 0.5
    t0, t1, dt0 = (1.0, 0.0, -0.125) if backward else (0.0, 1.0, 0.125)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "dt0": dt0,
        "y0": jnp.array([1.0, -0.5]),
        "args": -0.3,
        "saveat": diffrax.SaveAt(t0=True, t1=True, dense=True),
        "max_steps": 8,
    }
    geo = diffrax.diffeqsolve(geo_terms, solver, **kwargs)
    ref = diffrax.diffeqsolve(ref_terms, diffrax.EulerHeun(), **kwargs)
    np.testing.assert_allclose(geo.ys, ref.ys, atol=1e-6)
    np.testing.assert_allclose(geo.evaluate(0.37), ref.evaluate(0.37), atol=1e-6)


def test_geometric_euler_heun_requires_drift_and_diffusion():
    term = GeometricTerm(lambda t, y, args: -y, Euclidean())
    with pytest.raises(ValueError, match="Terms are not compatible"):
        diffrax.diffeqsolve(
            term, GeometricEulerHeun(), t0=0.0, t1=1.0, dt0=0.1, y0=jnp.ones(1)
        )
