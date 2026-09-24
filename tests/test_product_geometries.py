import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from georax import (
    CFEES25,
    CFEES27,
    CG2,
    SO,
    SPD,
    SRKMK,
    Euclidean,
    GeometricTerm,
    Product,
    Sphere,
    TangentTorus,
    Torus,
)

jax.config.update("jax_enable_x64", True)


def test_tangent_torus_is_an_exact_additive_product():
    geometry = TangentTorus(2)
    assert isinstance(geometry, Product)
    assert isinstance(geometry.factors[0], Torus)
    assert isinstance(geometry.factors[1], Euclidean)
    x = geometry.pack_state(jnp.array([3.0, -3.0]), jnp.array([12.0, -14.0]))
    a = geometry.pack_coordinates(jnp.array([0.5, -0.5]), jnp.array([2.0, -3.0]))
    chart = geometry.select_chart(5)
    theta, omega = geometry.unpack_state(geometry.apply_increment(x, a, chart))
    np.testing.assert_allclose(theta, jnp.array([3.5 - 2 * jnp.pi, -3.5 + 2 * jnp.pi]))
    np.testing.assert_array_equal(omega, jnp.array([14.0, -17.0]))
    np.testing.assert_allclose(
        jax.jacfwd(lambda a: chart.apply(x, a, geometry))(a), jnp.eye(4)
    )
    np.testing.assert_array_equal(chart.inverse_differential(x, a, a, geometry), a)
    assert jax.tree.leaves(chart) == []


def test_mixed_nested_product_charts_and_pullbacks():
    geometry = Product(Product(SPD(2), Torus(2)), SO(3), Sphere(4))
    inner = geometry.factors[0]
    x = geometry.pack_state(
        inner.pack_state(jnp.diag(jnp.array([1.0, 2.0])), jnp.array([0.1, 3.0])),
        jnp.eye(3),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
    )
    a = jnp.linspace(-0.15, 0.2, geometry.coordinate_shape[0])
    b = jnp.linspace(0.2, -0.3, geometry.coordinate_shape[0])
    chart = geometry.select_chart(4)
    delta = chart.inverse_differential(x, a, b, geometry)
    y, tangent = jax.jvp(
        lambda a: geometry.apply_increment(x, a, chart), (a,), (delta,)
    )
    np.testing.assert_allclose(tangent, geometry.detrivialise(y, b), atol=1e-12)
    np.testing.assert_allclose(
        geometry.detrivialise(y, geometry.trivialise(y, tangent)), tangent, atol=1e-12
    )
    pullback = geometry.select_pullback_chart(5)
    assert [c.order for c in pullback.charts] == [5, 2, 2]
    assert jax.tree.leaves(pullback) == []
    spd, theta = inner.unpack_state(geometry.unpack_state(jnp.stack([x, y]))[0])
    assert spd.shape == (2, 2, 2) and theta.shape == (2, 2)
    with pytest.raises(ValueError, match="Specify a Euclidean shape"):
        Product(Torus(2), Euclidean())
    with pytest.raises(ValueError, match="Factor array"):
        TangentTorus(2).pack_state(jnp.zeros(2), jnp.zeros(3))


def test_sphere_batched_rotation_conventions_and_pullback():
    geometry = Sphere(4)
    expected = jnp.array([[0, -1, -2, -4], [1, 0, -3, -5], [2, 3, 0, -6], [4, 5, 6, 0]])
    np.testing.assert_array_equal(
        geometry._coords_to_alg(jnp.arange(1.0, 7.0)), expected
    )
    x = jnp.eye(4).reshape(2, 2, 4)
    a = jnp.linspace(-0.2, 0.1, 24).reshape(2, 2, 6)
    b = jnp.linspace(0.3, -0.4, 24).reshape(2, 2, 6)
    chart = geometry.select_pullback_chart(5)
    assert chart.order == 2
    assert geometry.zero_coordinates(x).shape == a.shape
    delta = jax.jit(lambda a, b: chart.inverse_differential(x, a, b, geometry))(a, b)
    y, tangent = jax.jvp(lambda a: chart.apply(x, a, geometry), (a,), (delta,))
    np.testing.assert_allclose(jnp.linalg.norm(y, axis=-1), 1, atol=1e-13)
    np.testing.assert_allclose(tangent, geometry.detrivialise(y, b), atol=1e-13)
    np.testing.assert_allclose(
        geometry.detrivialise(y, geometry.trivialise(y, tangent)), tangent, atol=1e-13
    )
    # Check the sign of the bracket for the left SO(n) action.
    expected_bracket = (
        jax.jvp(
            lambda y: geometry.detrivialise(y, b), (x,), (geometry.detrivialise(x, a),)
        )[1]
        - jax.jvp(
            lambda y: geometry.detrivialise(y, a), (x,), (geometry.detrivialise(x, b),)
        )[1]
    )
    np.testing.assert_allclose(
        geometry.detrivialise(x, geometry.frame_bracket(x, a, b)),
        expected_bracket,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "solver", [CG2(), CFEES25(), CFEES27()], ids=lambda s: type(s).__name__
)
def test_product_saved_observations_and_gradients(solver):
    geometry = Product(TangentTorus(2), SPD(2), SO(2), Sphere(3))
    tangent_torus = geometry.factors[0]
    y0 = geometry.pack_state(
        tangent_torus.pack_state(jnp.array([3.1, -3.1]), jnp.array([4.0, -5.0])),
        jnp.eye(2),
        jnp.eye(2),
        jnp.array([1.0, 0.0, 0.0]),
    )

    def coeffs(t, y, weight):
        rotor, spd, rotation, sphere = geometry.unpack_state(y)
        theta, omega = tangent_torus.unpack_state(rotor)
        return geometry.pack_coordinates(
            tangent_torus.pack_coordinates(
                omega, weight * jnp.sin(theta[::-1] - theta) - 0.2 * omega
            ),
            weight * jnp.array([0.1, jnp.sin(theta[0]), 0.2 * rotation[0, 0]]),
            weight * jnp.trace(spd)[None],
            weight * (jnp.array([0.7, -0.2, 0.4]) + 0.1 * sphere[::-1]),
        )

    def solve(weight, adjoint):
        return diffrax.diffeqsolve(
            GeometricTerm(coeffs, geometry),
            solver,
            t0=0.0,
            t1=0.2,
            dt0=0.025,
            y0=y0,
            args=weight,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(ts=jnp.array([0.013, 0.084, 0.177, 0.2])),
            max_steps=8,
        ).ys

    samples = solve(0.7, diffrax.RecursiveCheckpointAdjoint())
    rotor, spd, rotation, sphere = geometry.unpack_state(samples)
    theta, omega = tangent_torus.unpack_state(rotor)
    assert jnp.all(theta >= -jnp.pi) and jnp.all(theta < jnp.pi)
    assert jnp.all(jnp.abs(omega) > jnp.pi)
    assert jnp.all(jnp.linalg.eigvalsh(spd) > 0)
    np.testing.assert_allclose(spd, jnp.swapaxes(spd, -1, -2), atol=1e-12)
    np.testing.assert_allclose(
        jnp.swapaxes(rotation, -1, -2) @ rotation,
        jnp.broadcast_to(jnp.eye(2), rotation.shape),
        atol=2e-12,
    )
    np.testing.assert_allclose(jnp.linalg.det(rotation), 1, atol=2e-12)
    np.testing.assert_allclose(jnp.linalg.norm(sphere, axis=-1), 1, atol=2e-12)

    def loss(weight, adjoint):
        return jnp.sin(solve(weight, adjoint)).sum()

    forward = jax.jacfwd(lambda w: loss(w, diffrax.DirectAdjoint()))(0.7)
    reverse = jax.grad(lambda w: loss(w, diffrax.RecursiveCheckpointAdjoint()))(0.7)
    eps = 1e-5
    finite_difference = (
        loss(0.7 + eps, diffrax.DirectAdjoint())
        - loss(0.7 - eps, diffrax.DirectAdjoint())
    ) / (2 * eps)
    assert jnp.isfinite(forward) and abs(forward) > 1e-6
    np.testing.assert_allclose(forward, finite_difference, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(reverse, forward, rtol=1e-10, atol=1e-11)
    if isinstance(solver, (CFEES25, CFEES27)):
        reversible = jax.grad(lambda w: loss(w, diffrax.ReversibleAdjoint()))(0.7)
        np.testing.assert_allclose(reversible, reverse, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("backward", [False, True])
def test_batched_sphere_srkmk_observations_and_gradients(backward):
    geometry = Sphere(4)
    y0 = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.8, 0.0]])
    path = diffrax.VirtualBrownianTree(
        0.0, 0.2, 1e-4, (2,), jax.random.key(51), levy_area=diffrax.SpaceTimeLevyArea
    )
    drift = GeometricTerm(
        lambda t, y, weight: (
            weight * (jnp.linspace(-0.2, 0.3, 6) + 0.1 * y[..., 0, None])
        ),
        geometry,
    )
    diffusion = diffrax.ControlTerm(
        lambda t, y, weight: (
            0.15 * weight * (1 + 0.1 * y[..., 0, None, None]) * jnp.eye(6)[:, :2]
        ),
        path,
    )
    terms = diffrax.MultiTerm(drift, diffusion)
    solver = SRKMK(diffrax.GeneralShARK())
    t0, t1, dt = (0.2, 0.0, -0.05) if backward else (0.0, 0.2, 0.05)
    times = jnp.array([0.0, 0.035, 0.16, 0.2])

    def solve(weight, adjoint):
        return diffrax.diffeqsolve(
            terms,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=y0,
            args=weight,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(ts=times[::-1] if backward else times),
            max_steps=4,
        ).ys

    samples = solve(0.7, diffrax.RecursiveCheckpointAdjoint())
    np.testing.assert_allclose(jnp.linalg.norm(samples, axis=-1), 1, atol=2e-12)

    def loss(weight, adjoint):
        return jnp.sin(solve(weight, adjoint)).sum()

    forward = jax.jacfwd(lambda w: loss(w, diffrax.DirectAdjoint()))(0.7)
    reverse = jax.grad(lambda w: loss(w, diffrax.RecursiveCheckpointAdjoint()))(0.7)
    eps = 1e-5
    finite_difference = (
        loss(0.7 + eps, diffrax.DirectAdjoint())
        - loss(0.7 - eps, diffrax.DirectAdjoint())
    ) / (2 * eps)
    assert jnp.isfinite(forward) and abs(forward) > 1e-6
    np.testing.assert_allclose(forward, finite_difference, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(reverse, forward, rtol=1e-10, atol=1e-11)
