from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from georax import (
    SO,
    SPD,
    Euclidean,
    GeometricEuler,
    GeometricEulerHeun,
    GeometricTerm,
)


def _brownian(shape, key):
    return diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        tol=1e-3,
        shape=shape,
        key=jax.random.key(key),
    )


def test_geometric_euler_preserves_so3() -> None:
    geometry = SO(3)
    term = GeometricTerm(
        lambda t, y, args: jnp.array([0.2, -0.1, 0.05]),
        geometry=geometry,
    )

    out = diffrax.diffeqsolve(
        term,
        GeometricEuler(),
        t0=0.0,
        t1=0.2,
        dt0=0.05,
        y0=jnp.eye(3),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
        throw=True,
    )
    assert out.ys is not None
    y1 = out.ys[0]

    assert bool(jnp.allclose(y1.T @ y1, jnp.eye(3), atol=1e-6))
    assert float(jnp.linalg.det(y1)) > 0.0


def test_geometric_euler_maruyama_preserves_so3() -> None:
    geometry = SO(3)
    terms = diffrax.MultiTerm(
        GeometricTerm(lambda t, y, args: jnp.zeros(3), geometry=geometry),
        diffrax.ControlTerm(
            lambda t, y, args: lx.DiagonalLinearOperator(jnp.array([0.1, 0.2, 0.3])),
            _brownian((3,), 0),
        ),
    )

    solver = GeometricEuler()
    assert isinstance(solver, diffrax.AbstractItoSolver)
    assert solver.strong_order(terms) == 0.5

    out = diffrax.diffeqsolve(
        terms,
        solver,
        t0=0.0,
        t1=0.2,
        dt0=0.05,
        y0=jnp.eye(3),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
        throw=True,
    )
    assert out.ys is not None
    y1 = out.ys[0]

    assert bool(jnp.allclose(y1.T @ y1, jnp.eye(3), atol=1e-6))
    assert float(jnp.linalg.det(y1)) > 0.0


def test_geometric_euler_matches_diffrax_euler_on_euclidean_geometry() -> None:
    def drift(t, y, args):
        del t, args
        return -0.2 * y

    def diffusion(t, y, args):
        del t, args
        return lx.DiagonalLinearOperator(0.1 + 0.05 * y)

    y0 = jnp.array([1.0, -0.5])
    bm = _brownian((2,), 1)
    terms_geo = diffrax.MultiTerm(
        GeometricTerm(drift, geometry=Euclidean()),
        diffrax.ControlTerm(diffusion, bm),
    )
    terms_ref = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, bm),
    )

    out_geo = diffrax.diffeqsolve(
        terms_geo,
        GeometricEuler(),
        t0=0.0,
        t1=0.5,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
        throw=True,
    )
    out_ref = diffrax.diffeqsolve(
        terms_ref,
        diffrax.Euler(),
        t0=0.0,
        t1=0.5,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
        throw=True,
    )

    assert out_geo.ys is not None
    assert out_ref.ys is not None
    assert bool(jnp.allclose(out_geo.ys[0], out_ref.ys[0], atol=1e-6))


@pytest.mark.parametrize("solver", [GeometricEuler(), GeometricEulerHeun()])
@pytest.mark.parametrize("geometry", [Euclidean(), SPD(1)])
def test_multiplicative_noise_converges_to_declared_interpretation(solver, geometry):
    # In the global coordinate x = log(y) on SPD(1), the factor 1/2 undoes
    # its infinitesimal congruence action. Both cases solve dx = sigma*x dW,
    # whose exact Itô and Stratonovich solutions are different.
    sigma = 0.4
    num_steps = 256
    increments = jax.random.normal(jax.random.key(58), (256, num_steps)) / jnp.sqrt(
        num_steps
    )
    paths = jnp.concatenate(
        (jnp.zeros((256, 1)), jnp.cumsum(increments, axis=1)), axis=1
    )
    curved = isinstance(geometry, SPD)
    y0 = jnp.array([[jnp.exp(0.5)]]) if curved else jnp.array([0.5])

    def coefficients(t, y, args):
        del t, args
        x = jnp.log(y[0, 0]) if curved else y[0]
        return jnp.reshape((0.5 if curved else 1.0) * sigma * x, (1, 1))

    errors = []
    is_ito = isinstance(solver, diffrax.AbstractItoSolver)
    exact = 0.5 * jnp.exp(sigma * paths[:, -1] - (0.5 * sigma**2 if is_ito else 0))
    other = 0.5 * jnp.exp(sigma * paths[:, -1] - (0 if is_ito else 0.5 * sigma**2))
    if curved:
        exact, other = jnp.exp(exact), jnp.exp(other)

    for steps in (16, 64, num_steps):

        def solve(path, steps=steps):
            control = diffrax.LinearInterpolation(
                jnp.linspace(0.0, 1.0, steps + 1),
                path[:: num_steps // steps, None],
            )
            terms = diffrax.MultiTerm(
                GeometricTerm(
                    lambda t, y, args: geometry.zero_coordinates(y), geometry
                ),
                diffrax.ControlTerm(coefficients, control),
            )
            result = diffrax.diffeqsolve(
                terms,
                solver,
                t0=0.0,
                t1=1.0,
                dt0=1 / steps,
                y0=y0,
                max_steps=steps + 2,
            )
            return result.ys[0].reshape(())

        ys = jax.jit(jax.vmap(solve))(paths)
        errors.append(float(jnp.sqrt(jnp.mean((ys - exact) ** 2))))

    # Coupled Brownian paths give a deterministic refinement check, while the
    # second target guards against converging to the wrong stochastic integral.
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.35 * errors[0]
    wrong_error = float(jnp.sqrt(jnp.mean((ys - other) ** 2)))
    assert errors[2] < 0.2 * wrong_error


def test_geometric_euler_spd_ambient_ito_drift_includes_chart_correction():
    # Zero frame drift and constant diffusion sigma/2 produce ambient drift
    # sigma**2*y/2, rather than the zero drift of ambient Euler--Maruyama.
    geometry = SPD(1)
    sigma = 0.4
    dt = 1e-3
    y0 = jnp.array([[2.0]])
    solver = GeometricEuler()

    def step(dw, drift):
        terms = diffrax.MultiTerm(
            GeometricTerm(lambda t, y, args: jnp.array([drift]), geometry),
            diffrax.ControlTerm(
                lambda t, y, args: jnp.array([[sigma / 2]]),
                diffrax.LinearInterpolation(jnp.array([0.0, dt]), dw[:, None]),
            ),
        )
        return solver.step(terms, 0.0, dt, y0, None, None, False)[0]

    def mean_increment(drift):
        plus = step(jnp.array([0.0, jnp.sqrt(dt)]), drift)
        minus = step(jnp.array([0.0, -jnp.sqrt(dt)]), drift)
        return (0.5 * (plus + minus) - y0) / dt

    np.testing.assert_allclose(mean_increment(0.0), 0.5 * sigma**2 * y0, atol=3e-4)
    np.testing.assert_allclose(mean_increment(-(sigma**2) / 4), 0.0, atol=3e-4)
