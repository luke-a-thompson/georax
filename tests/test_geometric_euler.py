from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax._solver.base import AbstractItoSolver

from georax import SO, Euclidean, GeometricEuler, GeometricTerm


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
    assert isinstance(solver, AbstractItoSolver)
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
