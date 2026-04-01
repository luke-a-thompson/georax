from __future__ import annotations

import jax.numpy as jnp
import pytest
from conftest import EuclideanOps
from diffrax import Heun, ODETerm

from georax import CG2, RKMK, GeometricTerm


def test_rkmk_rejects_non_erk_base_solver() -> None:
    with pytest.raises(TypeError):
        RKMK(CG2())


def test_rkmk_requires_lie_group_geometry() -> None:
    solver = RKMK(Heun())
    term = GeometricTerm(inner=ODETerm(lambda t, y, args: y), geometry=EuclideanOps())

    with pytest.raises(TypeError):
        solver.step(
            terms=term,
            t0=0.0,
            t1=1.0,
            y0=jnp.array([1.0]),
            args=None,
            solver_state=None,
            made_jump=False,
        )
