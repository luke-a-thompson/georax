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


# ── SO(3) accuracy terms (used by tests and docs examples) ────────────────────

_SO3_GEOMETRY = SO(3)
_SO3_NOISE_SCALE = jnp.array([0.12, 0.08, 0.05], dtype=jnp.float32)


def so3_body_omega(t: RealScalarLike) -> Array:
    t_array = jnp.asarray(t)
    return jnp.array(
        [
            0.8 + 0.45 * jnp.sin(0.7 * t_array),
            0.55 * jnp.cos(1.3 * t_array + 0.2),
            0.35 + 0.6 * jnp.sin(0.9 * t_array - 0.4),
        ]
    )


def so3_body_frame_coeffs(t: RealScalarLike) -> Array:
    omega = so3_body_omega(t)
    return jnp.array([-omega[2], omega[1], -omega[0]], dtype=omega.dtype)


def make_solver_accuracy_term() -> GeometricTerm:
    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        return _SO3_GEOMETRY.from_frame(R, so3_body_frame_coeffs(t))

    return GeometricTerm(inner=diffrax.ODETerm(vf), geometry=_SO3_GEOMETRY)


def make_solver_accuracy_sde_term(
    *,
    bm_tol: float = 1e-3,
    key: Array | None = None,
) -> GeometricTerm:
    if key is None:
        key = jax.random.key(0)

    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        tol=bm_tol,
        shape=(_SO3_GEOMETRY.lie_algebra_dimension,),
        key=key,
    )

    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        return _SO3_GEOMETRY.from_frame(R, so3_body_frame_coeffs(t))

    def diffusion_op(t: RealScalarLike, R: Array, args: Args):
        del t, R, args
        return lineax.DiagonalLinearOperator(_SO3_NOISE_SCALE)

    def coeffs_prod(t: RealScalarLike, R: Array, args: Args, control) -> Array:
        del R, args
        dt, dW = control
        return so3_body_frame_coeffs(t) * dt + _SO3_NOISE_SCALE * dW

    return GeometricTerm(
        inner=MultiTerm(
            ODETerm(vf),
            diffrax.ControlTerm(diffusion_op, brownian),
        ),
        geometry=_SO3_GEOMETRY,
        coeffs_prod_fn=coeffs_prod,
    )


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

# ── SO(N) MLP VF ──────────────────────────────────────────────────────────────

# Use a high-dimensional SO(N) benchmark so the manifold state and Lie algebra
# coordinates are large enough to expose solver-owned stage storage costs.
_SON_N = 46
_SON_DIM = _SON_N * (_SON_N - 1) // 2  # 1035, close to the Euclidean 1028 state
_SON_HIDDEN = 768
_SON_AMBIENT_DIM = _SON_N * _SON_N
_SON_NUM_LAYERS = 3
_son_keys = jax.random.split(jax.random.key(7), _SON_NUM_LAYERS)
_SON_W_IN = jax.random.normal(_son_keys[0], (_SON_HIDDEN, _SON_AMBIENT_DIM)) / jnp.sqrt(
    float(_SON_AMBIENT_DIM)
)
_SON_W_MID = tuple(
    jax.random.normal(key, (_SON_HIDDEN, _SON_HIDDEN)) / jnp.sqrt(float(_SON_HIDDEN))
    for key in _son_keys[1:-1]
)
_SON_W_OUT = jax.random.normal(_son_keys[-1], (_SON_DIM, _SON_HIDDEN)) / jnp.sqrt(
    float(_SON_HIDDEN)
)
_SON_PARAMS = (_SON_W_IN, _SON_W_MID, _SON_W_OUT)

# Precompute SO(N) skew basis so VFs don't depend on a specific geometry instance.
_SON_GEOMETRY = SO(_SON_N)
_SON_BASIS = _SON_GEOMETRY._basis


def _son_apply(weights, R):
    w_in, w_mid_layers, w_out = weights
    h = jnp.tanh(w_in @ R.flatten())
    for w_mid in w_mid_layers:
        h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)  # (dim so(n),) frame coords


def _son_ambient_from_frame(R, a):
    """Lift frame coords to ambient tangent R @ skew(a)."""
    omega = jnp.einsum("ijk,k->ij", _SON_BASIS, a)
    return R @ omega


def _son_vf_global(t, R, args):
    del t, args
    return _son_ambient_from_frame(R, _son_apply(_SON_PARAMS, R))


def _son_vf_args(t, R, args):
    del t
    return _son_ambient_from_frame(R, _son_apply(args, R))


# For the SDE, noise lives in the Lie algebra: dW ∈ R^dim maps to frame coords
# via constant scaling. We use coeffs_prod_fn to express this directly in frame
# coordinates, bypassing the need for a dim -> (N, N) linear operator.
_SON_NOISE_SCALE = jnp.full((_SON_DIM,), 0.05)
_bench_bm_son = diffrax.VirtualBrownianTree(
    t0=0.0, t1=1.0, tol=1e-3, shape=(_SON_DIM,), key=jax.random.key(1)
)


def _son_sde_coeffs_prod_global(t, R, args, control):
    dt, dW = control
    return _son_apply(_SON_PARAMS, R) * dt + _SON_NOISE_SCALE * dW


def _son_sde_coeffs_prod_args(t, R, args, control):
    dt, dW = control
    return _son_apply(args, R) * dt + _SON_NOISE_SCALE * dW


_BENCH_SON_ODE_TERM_GLOBAL = GeometricTerm(
    inner=ODETerm(_son_vf_global),
    geometry=_SON_GEOMETRY,
)
_BENCH_SON_ODE_TERM_ARGS = GeometricTerm(
    inner=ODETerm(_son_vf_args),
    geometry=_SON_GEOMETRY,
)
# The inner MultiTerm provides contr = (dt, dW); coeffs_prod_fn computes frame
# coefficients directly, so the ControlTerm VF is never evaluated.
_BENCH_SON_SDE_TERM_GLOBAL = GeometricTerm(
    inner=MultiTerm(
        ODETerm(_son_vf_global),
        diffrax.ControlTerm(
            lambda t, R, args: lineax.DiagonalLinearOperator(_SON_NOISE_SCALE),
            _bench_bm_son,
        ),
    ),
    geometry=_SON_GEOMETRY,
    coeffs_prod_fn=_son_sde_coeffs_prod_global,
)
_BENCH_SON_SDE_TERM_ARGS = GeometricTerm(
    inner=MultiTerm(
        ODETerm(_son_vf_args),
        diffrax.ControlTerm(
            lambda t, R, args: lineax.DiagonalLinearOperator(_SON_NOISE_SCALE),
            _bench_bm_son,
        ),
    ),
    geometry=_SON_GEOMETRY,
    coeffs_prod_fn=_son_sde_coeffs_prod_args,
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
        f"ode/so{_SON_N}/global",
        _BENCH_SON_ODE_TERM_GLOBAL,
        jnp.eye(_SON_N, dtype=jnp.float32),
        None,
        "y0",
    ),
    BenchCase(
        f"ode/so{_SON_N}/args",
        _BENCH_SON_ODE_TERM_ARGS,
        jnp.eye(_SON_N, dtype=jnp.float32),
        _SON_PARAMS,
        "args",
    ),
    BenchCase(
        f"sde/so{_SON_N}/global",
        _BENCH_SON_SDE_TERM_GLOBAL,
        jnp.eye(_SON_N, dtype=jnp.float32),
        None,
        "y0",
    ),
    BenchCase(
        f"sde/so{_SON_N}/args",
        _BENCH_SON_SDE_TERM_ARGS,
        jnp.eye(_SON_N, dtype=jnp.float32),
        _SON_PARAMS,
        "args",
    ),
]
