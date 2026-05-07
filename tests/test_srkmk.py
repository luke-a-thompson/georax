from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx

from georax import SO, Euclidean, GeometricTerm, SRKMK


class DiagonalControlTerm(diffrax.AbstractTerm):
    diagonal: jax.Array
    control: diffrax.AbstractPath

    def __init__(self, diagonal, control):
        self.diagonal = diagonal
        self.control = control

    def vf(self, t, y, args):
        del t, args
        return self.diagonal + jnp.zeros_like(y)

    def contr(self, t0, t1, **kwargs):
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf, control):
        return vf * control


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


def test_srkmk_matches_base_srk_on_euclidean_geometry() -> None:
    def drift(t, y, args):
        del t, args
        return -0.2 * y

    def diffusion(t, y, args):
        del t, y, args
        return lx.DiagonalLinearOperator(jnp.array([0.3, 0.7]))

    y0 = jnp.array([1.0, -0.5])
    terms_geo = diffrax.MultiTerm(
        GeometricTerm(drift, geometry=Euclidean()),
        diffrax.ControlTerm(diffusion, _brownian((2,), 1)),
    )
    terms_ref = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, _brownian((2,), 1)),
    )

    srk = diffrax.SRA1()
    solver = SRKMK(srk, additive_after_pullback=True)
    assert solver.strong_order(terms_geo) == 1.5

    out_geo = diffrax.diffeqsolve(
        terms_geo,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=16,
    )
    out_ref = diffrax.diffeqsolve(
        terms_ref,
        srk,
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=16,
    )

    assert out_geo.ys is not None
    assert out_ref.ys is not None
    assert bool(jnp.allclose(out_geo.ys[0], out_ref.ys[0], atol=1e-6))


def test_srkmk_uses_diffusion_term_prod_on_euclidean_geometry() -> None:
    def drift(t, y, args):
        del t, args
        return -0.1 * y

    y0 = jnp.array([1.0, -0.5])
    diagonal = jnp.array([0.3, 0.7])
    terms_geo = diffrax.MultiTerm(
        GeometricTerm(drift, geometry=Euclidean()),
        DiagonalControlTerm(diagonal, _brownian((2,), 4)),
    )
    terms_ref = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        DiagonalControlTerm(diagonal, _brownian((2,), 4)),
    )

    srk = diffrax.GeneralShARK()
    out_geo = diffrax.diffeqsolve(
        terms_geo,
        SRKMK(srk),
        t0=0.0,
        t1=0.5,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
    )
    out_ref = diffrax.diffeqsolve(
        terms_ref,
        srk,
        t0=0.0,
        t1=0.5,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=8,
    )

    assert out_geo.ys is not None
    assert out_ref.ys is not None
    assert bool(jnp.allclose(out_geo.ys[0], out_ref.ys[0], atol=1e-6))


def test_additive_srk_requires_pullback_additivity_assumption() -> None:
    terms = diffrax.MultiTerm(
        GeometricTerm(lambda t, y, args: jnp.zeros_like(y), geometry=Euclidean()),
        diffrax.ControlTerm(lambda t, y, args: jnp.eye(2), _brownian((2,), 2)),
    )

    solver = SRKMK(diffrax.SRA1())
    try:
        solver.init(terms, 0.0, 0.1, jnp.ones(2), None)
    except TypeError as exc:
        assert "additive noise" in str(exc)
    else:
        raise AssertionError("Expected additive SRK to require an explicit assumption.")
