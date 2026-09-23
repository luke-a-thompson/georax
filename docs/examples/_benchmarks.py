from __future__ import annotations

from typing import Any, Literal, NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import AbstractTerm, ControlTerm, MultiTerm
from jaxtyping import PyTree

from georax import (
    CFEES25,
    CFEES27,
    CG2,
    CG4,
    RKMK,
    SO,
    SRKMK,
    Euclidean,
    GeometricTerm,
)

BENCH_SOLVERS = [("cg2", CG2), ("cg4", CG4), ("cfees25", CFEES25), ("cfees27", CFEES27)]
ODE_BENCH_SOLVERS = [*BENCH_SOLVERS, ("rkmk_tsit5", lambda: RKMK(diffrax.Tsit5()))]
SDE_BENCH_SOLVERS = [
    *BENCH_SOLVERS,
    ("srkmk_gen_shark", lambda: SRKMK(diffrax.GeneralShARK())),
]


class BenchCase(NamedTuple):
    name: str
    term: AbstractTerm
    y0: Any
    args: PyTree | None
    grad_target: Literal["y0", "args"]


def bench_solvers_for_case(case: BenchCase):
    return SDE_BENCH_SOLVERS if case.name.startswith("sde/") else ODE_BENCH_SOLVERS


def _mlp(weights, y):
    w_in, w_mid_layers, w_out = weights
    h = jnp.tanh(w_in @ y)
    for w_mid in w_mid_layers:
        h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)


def _weights(key, input_dim: int, hidden_dim: int, output_dim: int):
    keys = jax.random.split(key, 3)
    return (
        jax.random.normal(keys[0], (hidden_dim, input_dim)) / jnp.sqrt(input_dim),
        (jax.random.normal(keys[1], (hidden_dim, hidden_dim)) / jnp.sqrt(hidden_dim),),
        jax.random.normal(keys[2], (output_dim, hidden_dim)) / jnp.sqrt(hidden_dim),
    )


def _term_set(geometry, vf_global, vf_args, diffusion, brownian):
    return (
        GeometricTerm(vf_global, geometry),
        GeometricTerm(vf_args, geometry),
        MultiTerm(GeometricTerm(vf_global, geometry), ControlTerm(diffusion, brownian)),
        MultiTerm(GeometricTerm(vf_args, geometry), ControlTerm(diffusion, brownian)),
    )


_EUCLIDEAN_DIM = 32
_EUCLIDEAN_PARAMS = _weights(jax.random.key(42), _EUCLIDEAN_DIM, 16, _EUCLIDEAN_DIM)


def _euclidean_global(t, y, args):
    del t, args
    return _mlp(_EUCLIDEAN_PARAMS, y)


def _euclidean_args(t, y, args):
    del t
    return _mlp(args, y)


def _euclidean_diffusion(t, y, args):
    del t, args
    return lx.DiagonalLinearOperator(0.1 * y)


_EUCLIDEAN_TERMS = _term_set(
    Euclidean(),
    _euclidean_global,
    _euclidean_args,
    _euclidean_diffusion,
    diffrax.VirtualBrownianTree(
        0.0,
        1.0,
        1e-3,
        (_EUCLIDEAN_DIM,),
        jax.random.key(0),
        levy_area=diffrax.SpaceTimeLevyArea,
    ),
)

_SO_N = 32
_SO_AMBIENT_DIM = _SO_N**2
_SO_COORDINATE_DIM = _SO_N * (_SO_N - 1) // 2
_SO_PARAMS = _weights(jax.random.key(7), _SO_AMBIENT_DIM, 768, _SO_COORDINATE_DIM)
_SO_NOISE_SCALE = jnp.full((_SO_COORDINATE_DIM,), 0.05)


def _so_global(t, state, args):
    del t, args
    return _mlp(_SO_PARAMS, state.flatten())


def _so_args(t, state, args):
    del t
    return _mlp(args, state.flatten())


def _so_diffusion(t, state, args):
    del t, state, args
    return lx.DiagonalLinearOperator(_SO_NOISE_SCALE)


_SO_TERMS = _term_set(
    SO(_SO_N),
    _so_global,
    _so_args,
    _so_diffusion,
    diffrax.VirtualBrownianTree(
        0.0,
        1.0,
        1e-3,
        (_SO_COORDINATE_DIM,),
        jax.random.key(1),
        levy_area=diffrax.SpaceTimeLevyArea,
    ),
)


def _bench_group(name, y0, terms, params):
    ode_global, ode_args, sde_global, sde_args = terms
    return [
        BenchCase(f"ode/{name}/global", ode_global, y0, None, "y0"),
        BenchCase(f"ode/{name}/args", ode_args, y0, params, "args"),
        BenchCase(f"sde/{name}/global", sde_global, y0, None, "y0"),
        BenchCase(f"sde/{name}/args", sde_args, y0, params, "args"),
    ]


BENCH_CASES = [
    *_bench_group(
        "euclidean",
        jnp.ones((_EUCLIDEAN_DIM,), dtype=jnp.float32),
        _EUCLIDEAN_TERMS,
        _EUCLIDEAN_PARAMS,
    ),
    *_bench_group(
        f"so{_SO_N}",
        jnp.eye(_SO_N, dtype=jnp.float32),
        _SO_TERMS,
        _SO_PARAMS,
    ),
]
