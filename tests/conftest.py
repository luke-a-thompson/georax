from __future__ import annotations

from typing import Any, Literal, NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import lineax
from diffrax import MultiTerm, ODETerm
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array, PyTree

from georax import CFEES25, CFEES27, CG2, CG4, SO, Euclidean, GeometricTerm

# ── Solvers ────────────────────────────────────────────────────────────────────

SOLVERS = [
    ("cg2", CG2),
    ("cg4", CG4),
    ("cfees25", CFEES25),
    ("cfees27", CFEES27),
]

BENCH_SOLVERS = SOLVERS


# ── BenchCase ──────────────────────────────────────────────────────────────────


class BenchCase(NamedTuple):
    name: str
    term: GeometricTerm
    y0: Any
    args: PyTree | None
    grad_target: Literal["y0", "args"]


# ── SO(3) accuracy term (used by test_solver_accuracy.py) ─────────────────────


def make_solver_accuracy_term() -> GeometricTerm:
    def omega_body(t: RealScalarLike) -> Array:
        t_array = jnp.asarray(t)
        return jnp.array(
            [
                0.8 + 0.45 * jnp.sin(0.7 * t_array),
                0.55 * jnp.cos(1.3 * t_array + 0.2),
                0.35 + 0.6 * jnp.sin(0.9 * t_array - 0.4),
            ]
        )

    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        omega = omega_body(t)
        skew = jnp.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ]
        )
        return R @ skew

    return GeometricTerm(inner=diffrax.ODETerm(vf), geometry=SO(3))


# ── Euclidean MLP ──────────────────────────────────────────────────────────────

# A small MLP as the ODE drift. Realistic per-stage activation retention cost
# (many intermediate tensors per vf_prod eval) — matters for reverse-mode
# memory comparisons, where a cheap `y**3` VF would be dominated by solver-
# independent scratch and wouldn't reveal stage-count scaling.
_BENCH_DIM = 1028
_BENCH_HIDDEN = 768
_BENCH_NUM_LAYERS = 3
_bench_keys = jax.random.split(jax.random.key(42), _BENCH_NUM_LAYERS)
_BENCH_W_IN = jax.random.normal(_bench_keys[0], (_BENCH_HIDDEN, _BENCH_DIM)) / jnp.sqrt(
    _BENCH_DIM
)
_BENCH_W_MID = tuple(
    jax.random.normal(key, (_BENCH_HIDDEN, _BENCH_HIDDEN)) / jnp.sqrt(_BENCH_HIDDEN)
    for key in _bench_keys[1:-1]
)
_BENCH_W_OUT = jax.random.normal(
    _bench_keys[-1], (_BENCH_DIM, _BENCH_HIDDEN)
) / jnp.sqrt(_BENCH_HIDDEN)
_BENCH_PARAMS = (_BENCH_W_IN, _BENCH_W_MID, _BENCH_W_OUT)


def _bench_apply(weights, y):
    w_in, w_mid_layers, w_out = weights
    h = jnp.tanh(w_in @ y)
    for w_mid in w_mid_layers:
        h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)


def _bench_vf_global(t, y, args):
    del t, args
    return _bench_apply(_BENCH_PARAMS, y)


def _bench_vf_args(t, y, args):
    del t
    return _bench_apply(args, y)


_bench_bm_euclidean = diffrax.VirtualBrownianTree(
    t0=0.0, t1=1.0, tol=1e-3, shape=(_BENCH_DIM,), key=jax.random.key(0)
)

_BENCH_EUCLIDEAN_ODE_TERM = GeometricTerm(
    inner=ODETerm(_bench_vf_global), geometry=Euclidean()
)
_BENCH_EUCLIDEAN_ODE_TERM_ARGS = GeometricTerm(
    inner=ODETerm(_bench_vf_args), geometry=Euclidean()
)
_BENCH_EUCLIDEAN_SDE_TERM = GeometricTerm(
    inner=MultiTerm(
        ODETerm(_bench_vf_global),
        diffrax.ControlTerm(
            lambda t, y, args: lineax.DiagonalLinearOperator(0.1 * y),
            _bench_bm_euclidean,
        ),
    ),
    geometry=Euclidean(),
)
_BENCH_EUCLIDEAN_SDE_TERM_ARGS = GeometricTerm(
    inner=MultiTerm(
        ODETerm(_bench_vf_args),
        diffrax.ControlTerm(
            lambda t, y, args: lineax.DiagonalLinearOperator(0.1 * y),
            _bench_bm_euclidean,
        ),
    ),
    geometry=Euclidean(),
)

# ── SO(3) MLP VF ──────────────────────────────────────────────────────────────

# A small MLP mapping the flattened rotation matrix R^9 to frame coordinates
# R^3. The ambient tangent is reconstructed via R @ skew(a) to stay in T_R SO(3).
# Using a matrix VF (rather than coeffs_fn) exercises the full geometry pipeline
# including to_frame projection.
_SO3_HIDDEN = 64
_so3_keys = jax.random.split(jax.random.key(7), 3)
_SO3_W_IN = jax.random.normal(_so3_keys[0], (_SO3_HIDDEN, 9)) / jnp.sqrt(9.0)
_SO3_W_MID = jax.random.normal(_so3_keys[1], (_SO3_HIDDEN, _SO3_HIDDEN)) / jnp.sqrt(
    float(_SO3_HIDDEN)
)
_SO3_W_OUT = jax.random.normal(_so3_keys[2], (3, _SO3_HIDDEN)) / jnp.sqrt(
    float(_SO3_HIDDEN)
)
_SO3_PARAMS = (_SO3_W_IN, _SO3_W_MID, _SO3_W_OUT)

# Precompute SO(3) skew basis so VFs don't depend on a specific geometry instance.
_SO3_BASIS = SO(3)._basis  # (3, 3, 3): basis[:, :, k] is the k-th skew basis matrix


def _so3_apply(weights, R):
    w_in, w_mid, w_out = weights
    h = jnp.tanh(w_in @ R.flatten())
    h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)  # (3,) frame coords


def _so3_ambient_from_frame(R, a):
    """Lift frame coords (3,) to ambient tangent (3, 3) = R @ skew(a)."""
    omega = jnp.einsum("ijk,k->ij", _SO3_BASIS, a)
    return R @ omega


def _so3_vf_global(t, R, args):
    del t, args
    return _so3_ambient_from_frame(R, _so3_apply(_SO3_PARAMS, R))


def _so3_vf_args(t, R, args):
    del t
    return _so3_ambient_from_frame(R, _so3_apply(args, R))


# For the SDE, noise lives in the Lie algebra: dW ∈ R^3 maps to frame coords
# via constant scaling. We use coeffs_prod_fn to express this directly in frame
# coordinates, bypassing the need for a (3,) → (3, 3) linear operator.
_SO3_NOISE_SCALE = jnp.full((3,), 0.05)
_bench_bm_so3 = diffrax.VirtualBrownianTree(
    t0=0.0, t1=1.0, tol=1e-3, shape=(3,), key=jax.random.key(1)
)


def _so3_sde_coeffs_prod_global(t, R, args, control):
    dt, dW = control
    return _so3_apply(_SO3_PARAMS, R) * dt + _SO3_NOISE_SCALE * dW


def _so3_sde_coeffs_prod_args(t, R, args, control):
    dt, dW = control
    return _so3_apply(args, R) * dt + _SO3_NOISE_SCALE * dW


_BENCH_SO3_ODE_TERM_GLOBAL = GeometricTerm(
    inner=ODETerm(_so3_vf_global),
    geometry=SO(3),
)
_BENCH_SO3_ODE_TERM_ARGS = GeometricTerm(
    inner=ODETerm(_so3_vf_args),
    geometry=SO(3),
)
# The inner MultiTerm provides contr = (dt, dW); coeffs_prod_fn computes frame
# coefficients directly, so the ControlTerm VF is never evaluated.
_BENCH_SO3_SDE_TERM_GLOBAL = GeometricTerm(
    inner=MultiTerm(
        ODETerm(_so3_vf_global),
        diffrax.ControlTerm(
            lambda t, R, args: lineax.DiagonalLinearOperator(_SO3_NOISE_SCALE),
            _bench_bm_so3,
        ),
    ),
    geometry=SO(3),
    coeffs_prod_fn=_so3_sde_coeffs_prod_global,
)
_BENCH_SO3_SDE_TERM_ARGS = GeometricTerm(
    inner=MultiTerm(
        ODETerm(_so3_vf_args),
        diffrax.ControlTerm(
            lambda t, R, args: lineax.DiagonalLinearOperator(_SO3_NOISE_SCALE),
            _bench_bm_so3,
        ),
    ),
    geometry=SO(3),
    coeffs_prod_fn=_so3_sde_coeffs_prod_args,
)

# ── Benchmark Cases ────────────────────────────────────────────────────────────

# `global` keeps MLP weights baked into HLO, differentiating wrt `y0`.
# `args` threads weights through args so reverse-mode must retain the full
# activation path needed for parameter gradients.
BENCH_CASES = [
    BenchCase(
        "ode/euclidean/global",
        _BENCH_EUCLIDEAN_ODE_TERM,
        jnp.ones((_BENCH_DIM,), dtype=jnp.float32),
        None,
        "y0",
    ),
    BenchCase(
        "ode/euclidean/args",
        _BENCH_EUCLIDEAN_ODE_TERM_ARGS,
        jnp.ones((_BENCH_DIM,), dtype=jnp.float32),
        _BENCH_PARAMS,
        "args",
    ),
    BenchCase(
        "sde/euclidean/global",
        _BENCH_EUCLIDEAN_SDE_TERM,
        jnp.ones((_BENCH_DIM,), dtype=jnp.float32),
        None,
        "y0",
    ),
    BenchCase(
        "sde/euclidean/args",
        _BENCH_EUCLIDEAN_SDE_TERM_ARGS,
        jnp.ones((_BENCH_DIM,), dtype=jnp.float32),
        _BENCH_PARAMS,
        "args",
    ),
    BenchCase(
        "ode/so3/global",
        _BENCH_SO3_ODE_TERM_GLOBAL,
        jnp.eye(3, dtype=jnp.float32),
        None,
        "y0",
    ),
    BenchCase(
        "ode/so3/args",
        _BENCH_SO3_ODE_TERM_ARGS,
        jnp.eye(3, dtype=jnp.float32),
        _SO3_PARAMS,
        "args",
    ),
    BenchCase(
        "sde/so3/global",
        _BENCH_SO3_SDE_TERM_GLOBAL,
        jnp.eye(3, dtype=jnp.float32),
        None,
        "y0",
    ),
    BenchCase(
        "sde/so3/args",
        _BENCH_SO3_SDE_TERM_ARGS,
        jnp.eye(3, dtype=jnp.float32),
        _SO3_PARAMS,
        "args",
    ),
]
