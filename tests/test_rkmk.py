from __future__ import annotations

import diffrax
import jax.numpy as jnp
import pytest
from conftest import EuclideanOps
from diffrax import Dopri5, Heun, ODETerm
from jaxtyping import Array

from georax import CG2, RKMK, GeometricTerm, LieGroup


class AdditiveLieGroup(LieGroup):
    def frame(self, x: Array) -> Array:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    def retraction(self, x: Array, v: Array) -> Array:
        return x + v

    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        del a
        return b


def _make_term(vf, geometry) -> GeometricTerm:
    return GeometricTerm(inner=ODETerm(vf), geometry=geometry)


def test_rkmk_is_transparently_adaptive() -> None:
    assert isinstance(RKMK(Dopri5()), diffrax.AbstractAdaptiveSolver)


def test_rkmk_rejects_non_erk_base_solver() -> None:
    with pytest.raises(TypeError, match="explicit Runge-Kutta"):
        RKMK(CG2())


def test_rkmk_requires_lie_group_geometry() -> None:
    solver = RKMK(Heun())
    term = _make_term(lambda t, y, args: y, EuclideanOps())

    with pytest.raises(TypeError, match="LieGroupOps"):
        solver.step(
            terms=term,
            t0=0.0,
            t1=1.0,
            y0=jnp.array([1.0]),
            args=None,
            solver_state=None,
            made_jump=False,
        )


def test_rkmk_matches_heun_on_additive_lie_group() -> None:
    solver = RKMK(Heun())

    def vf(t, y, args):
        del args
        return jnp.array([t + y[0]])

    y1, y_error, dense_info, _, _ = solver.step(
        terms=_make_term(vf, AdditiveLieGroup()),
        t0=0.0,
        t1=1.0,
        y0=jnp.array([1.0]),
        args=None,
        solver_state=None,
        made_jump=False,
    )

    assert bool(jnp.allclose(y1, jnp.array([3.0])))
    assert y_error is not None
    assert bool(jnp.allclose(y_error, jnp.array([-1.0])))
    assert dense_info["k"].shape == (2, 1)


def test_rkmk_diffeqsolve_uses_wrapped_interpolation() -> None:
    sol = diffrax.diffeqsolve(
        _make_term(lambda t, y, args: y, AdditiveLieGroup()),
        RKMK(Heun()),
        t0=0.0,
        t1=1.0,
        dt0=0.25,
        y0=jnp.array([1.0]),
        saveat=diffrax.SaveAt(ts=jnp.array([0.5, 1.0])),
    )

    assert sol.ys is not None
    assert sol.ys.shape == (2, 1)
