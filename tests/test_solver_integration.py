from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from georax import (
    CFEES25,
    CFEES27,
    CG2,
    CG4,
    RKMK,
    SO,
    SRKMK,
    Euclidean,
    GeometricEuler,
    GeometricTerm,
)

jax.config.update("jax_enable_x64", True)


def brownian(shape=(1,)):
    return diffrax.VirtualBrownianTree(
        0.0,
        1.0,
        1e-4,
        shape,
        jax.random.key(31),
        levy_area=diffrax.SpaceTimeLevyArea,
    )


@pytest.mark.parametrize("stochastic", [False, True])
def test_wrapped_solver_integrates_backward(stochastic):
    def drift(t, y, args):
        return jnp.full_like(y, 1.0 + t)

    geometric = GeometricTerm(drift, Euclidean())
    expected = -1.5
    if stochastic:
        path = brownian()
        diffusion = diffrax.ControlTerm(lambda t, y, args: jnp.ones((1, 1)), path)
        geometric = diffrax.MultiTerm(geometric, diffusion)
        solver = SRKMK(diffrax.GeneralShARK())
        expected = expected - path.evaluate(0.0, 1.0)[0]
    else:
        solver = RKMK(diffrax.Heun())
    kwargs = dict(t0=1.0, t1=0.0, dt0=-0.1, y0=jnp.zeros(1))
    actual = diffrax.diffeqsolve(geometric, solver, **kwargs).ys
    np.testing.assert_allclose(actual, [[expected]], atol=1e-12)


def test_additive_check_checks_the_pulled_back_diffusion():
    terms = diffrax.MultiTerm(
        GeometricTerm(lambda t, y, args: jnp.zeros(3), SO(3)),
        diffrax.ControlTerm(lambda t, y, args: jnp.eye(3), brownian((3,))),
    )
    with pytest.raises(ValueError, match="independent of y"):
        SRKMK(diffrax.SRA1(), additive_after_pullback=True).init(
            terms, 0.0, 0.1, jnp.eye(3), None
        )


SOLVERS = [
    GeometricEuler(),
    CG2(),
    CG4(),
    CFEES25(),
    CFEES27(),
    RKMK(diffrax.Heun()),
    SRKMK(diffrax.GeneralShARK()),
]


@pytest.mark.parametrize("solver", SOLVERS, ids=lambda s: type(s).__name__)
@pytest.mark.parametrize("backward", [False, True])
def test_saved_and_dense_so_states_stay_on_manifold(solver, backward):
    geometry = SO(3)
    term = GeometricTerm(lambda t, y, args: jnp.array([0.7 + t, 0.3, -0.2]), geometry)
    if isinstance(solver, SRKMK):
        term = diffrax.MultiTerm(
            term,
            diffrax.ControlTerm(lambda t, y, args: 0.1 * jnp.eye(3), brownian((3,))),
        )
    t0, t1, dt = (1.0, 0.0, -0.5) if backward else (0.0, 1.0, 0.5)
    times = jnp.array([t0, 0.75 if backward else 0.25, t1])
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=jnp.eye(3),
        saveat=diffrax.SaveAt(ts=times, dense=True),
        max_steps=4,
    )
    samples = jnp.concatenate((sol.ys, jax.vmap(sol.evaluate)(jnp.array([0.1, 0.6]))))
    for rotation in samples:
        np.testing.assert_allclose(rotation.T @ rotation, jnp.eye(3), atol=2e-12)
        np.testing.assert_allclose(jnp.linalg.det(rotation), 1.0, atol=2e-12)
    np.testing.assert_array_equal(sol.ys[0], jnp.eye(3))
    np.testing.assert_allclose(sol.evaluate(t1), sol.ys[-1], atol=1e-12)
    np.testing.assert_allclose(
        sol.evaluate(0.2, 0.8), sol.evaluate(0.8) - sol.evaluate(0.2)
    )


def test_solver_initialization_does_not_change_shared_geometry():
    geometry = SO(3)
    term = GeometricTerm(lambda t, y, args: jnp.array([0.7, 0.3, -0.2]), geometry)
    solver = CG4()
    y0 = jnp.eye(3)
    solver.init(term, 0.0, 0.5, y0, None)
    step = lambda: solver.step(term, 0.0, 0.5, y0, None, None, False)[0]
    before = step()
    GeometricEuler().init(term, 0.0, 0.5, y0, None)
    RKMK(diffrax.Heun()).init(term, 0.0, 0.5, y0, None)
    np.testing.assert_array_equal(step(), before)
    chart = geometry.select_chart(4)
    geometry.select_chart(2)
    assert chart.order == 4


class LinearField(eqx.Module):
    weight: jax.Array

    def __call__(self, t, y, args):
        return self.weight * y


@pytest.mark.parametrize("solver", [RKMK(diffrax.Heun()), CFEES25()])
def test_model_parameters_can_be_trained_through_geometric_term(solver):
    term = GeometricTerm(LinearField(jnp.array(0.2)), Euclidean())

    def loss(term):
        return diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=1.0, dt0=0.05, y0=jnp.ones(1)
        ).ys.sum()

    derivative = eqx.filter_jit(eqx.filter_grad(loss))(term).coeffs_fn.weight
    eps = 1e-5
    plus = eqx.tree_at(lambda t: t.coeffs_fn.weight, term, term.coeffs_fn.weight + eps)
    minus = eqx.tree_at(lambda t: t.coeffs_fn.weight, term, term.coeffs_fn.weight - eps)
    expected = (loss(plus) - loss(minus)) / (2 * eps)
    np.testing.assert_allclose(derivative, expected, rtol=1e-7)


@pytest.mark.parametrize("solver", [CFEES25(), CFEES27()])
def test_reversible_adjoint_remains_compatible(solver):
    term = GeometricTerm(lambda t, y, args: args * y, Euclidean())

    def loss(weight, adjoint):
        return diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.05,
            y0=jnp.ones(1),
            args=weight,
            adjoint=adjoint,
        ).ys.sum()

    expected = jax.grad(loss)(0.2, diffrax.RecursiveCheckpointAdjoint())
    actual = jax.grad(loss)(0.2, diffrax.ReversibleAdjoint())
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
