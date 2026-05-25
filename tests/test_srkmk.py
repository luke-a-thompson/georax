from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

from georax import SO, Euclidean, GeometricTerm, SRKMK


def _brownian(shape, key):
    return diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        tol=1e-3,
        shape=shape,
        key=jax.random.key(key),
        levy_area=diffrax.SpaceTimeLevyArea,
    )


def test_srkmk_accepts_controlterm_and_preserves_so3() -> None:
    geometry = SO(3)
    e0 = geometry._coords_to_alg(jnp.array([1.0, 0.0, 0.0]))
    e1 = geometry._coords_to_alg(jnp.array([0.0, 1.0, 0.0]))
    assert float(jnp.linalg.norm(e0 @ e1 - e1 @ e0)) > 0.0

    drift = GeometricTerm(lambda t, y, args: jnp.zeros(3), geometry=geometry)
    diffusion = diffrax.ControlTerm(
        lambda t, y, args: jnp.eye(3)[:, :2],
        _brownian((2,), 0),
    )
    terms = diffrax.MultiTerm(drift, diffusion)

    solver = SRKMK(diffrax.GeneralShARK())
    out = diffrax.diffeqsolve(
        terms,
        solver,
        t0=0.0,
        t1=0.1,
        dt0=0.1,
        y0=jnp.eye(3),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=2,
    )
    assert out.ys is not None
    y1 = out.ys[0]

    assert bool(jnp.allclose(y1.T @ y1, jnp.eye(3), atol=1e-6))
    assert float(jnp.linalg.det(y1)) > 0.0
    assert solver.strong_order(terms) == 0.5


def test_srkmk_slowrk_preserves_so6_torus() -> None:
    geometry = SO(6)
    generators = []
    cols = []
    for i, j in ((0, 1), (2, 3), (4, 5)):
        gen = jnp.zeros((6, 6)).at[i, j].set(1.0).at[j, i].set(-1.0)
        generators.append(gen)
        cols.append(geometry._alg_to_coords(gen))

    for i in range(3):
        for j in range(i):
            bracket = generators[i] @ generators[j] - generators[j] @ generators[i]
            assert bool(jnp.allclose(bracket, 0.0))

    diffusion_matrix = jnp.stack(cols, axis=1)
    terms = diffrax.MultiTerm(
        GeometricTerm(
            lambda t, y, args: jnp.zeros(geometry.coordinate_shape),
            geometry=geometry,
        ),
        diffrax.ControlTerm(lambda t, y, args: diffusion_matrix, _brownian((3,), 3)),
    )

    solver = SRKMK(diffrax.SlowRK())
    assert solver.strong_order(terms) == 1.5

    out = diffrax.diffeqsolve(
        terms,
        solver,
        t0=0.0,
        t1=0.2,
        dt0=0.05,
        y0=jnp.eye(6),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
    )
    assert out.ys is not None
    y1 = out.ys[0]
    assert bool(jnp.allclose(y1.T @ y1, jnp.eye(6), atol=1e-6))
    assert float(jnp.linalg.det(y1)) > 0.0


def test_additive_srk_requires_pullback_additivity_assumption() -> None:
    geometry = SO(3)
    terms = diffrax.MultiTerm(
        GeometricTerm(lambda t, y, args: jnp.zeros(3), geometry=geometry),
        diffrax.ControlTerm(lambda t, y, args: jnp.eye(3), _brownian((3,), 2)),
    )

    solver = SRKMK(diffrax.SRA1())
    try:
        solver.init(terms, 0.0, 0.1, jnp.eye(3), None)
    except TypeError as exc:
        assert "additive noise" in str(exc)
    else:
        raise AssertionError("Expected additive SRK to require an explicit assumption.")


def test_srkmk_rejects_shape_free_euclidean_geometry() -> None:
    terms = diffrax.MultiTerm(
        GeometricTerm(lambda t, y, args: jnp.zeros_like(y), geometry=Euclidean()),
        diffrax.ControlTerm(lambda t, y, args: jnp.eye(2), _brownian((2,), 5)),
    )

    try:
        SRKMK(diffrax.GeneralShARK()).init(terms, 0.0, 0.1, jnp.ones(2), None)
    except TypeError as exc:
        assert "fixed coordinate_shape" in str(exc)
    else:
        raise AssertionError("Expected SRKMK to reject shape-free Euclidean geometry.")
