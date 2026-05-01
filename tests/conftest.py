from __future__ import annotations

from typing import Any, Literal, NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
from diffrax import AbstractTerm, ControlTerm, MultiTerm, ODETerm
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array, PyTree

from georax import CFEES25, CFEES27, CG2, CG4, SO, SPD, Euclidean, GeometricTerm, SRKMK

# ── Solvers ────────────────────────────────────────────────────────────────────

SOLVERS = [
    ("cg2", CG2),
    # ("cg4", CG4),
    ("cfees25", CFEES25),
    ("cfees27", CFEES27),
]

BENCH_SOLVERS = SOLVERS


SDE_BENCH_SOLVERS = [
    *BENCH_SOLVERS,
    ("srkmk_gen_shark", lambda: SRKMK(diffrax.GeneralShARK())),
]


def bench_solvers_for_case(case: "BenchCase"):
    if case.name.startswith("sde/") and "/spd" not in case.name:
        return SDE_BENCH_SOLVERS
    return BENCH_SOLVERS


# ── BenchCase ──────────────────────────────────────────────────────────────────


class BenchCase(NamedTuple):
    name: str
    term: AbstractTerm
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
    def coeffs(t: RealScalarLike, R: Array, args: Args) -> Array:
        del R, args
        return so3_body_frame_coeffs(t)

    return GeometricTerm(coeffs, geometry=_SO3_GEOMETRY)


def make_solver_accuracy_ambient_term() -> ODETerm:
    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        return R @ _SO3_GEOMETRY._coords_to_alg(so3_body_frame_coeffs(t), dtype=R.dtype)

    return diffrax.ODETerm(vf)


def make_solver_accuracy_sde_term(
    *,
    bm_tol: float = 1e-3,
    key: Array | None = None,
) -> AbstractTerm:
    if key is None:
        key = jax.random.key(0)

    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=1.0,
        tol=bm_tol,
        shape=(_SO3_GEOMETRY.lie_algebra_dimension,),
        key=key,
    )

    def coeffs(t: RealScalarLike, R: Array, args: Args) -> Array:
        del R, args
        return so3_body_frame_coeffs(t)

    def diffusion(t: RealScalarLike, R: Array, args: Args):
        del t, R, args
        return lx.DiagonalLinearOperator(_SO3_NOISE_SCALE)

    return MultiTerm(
        GeometricTerm(coeffs, geometry=_SO3_GEOMETRY),
        ControlTerm(diffusion, brownian),
    )


# ── Euclidean MLP ──────────────────────────────────────────────────────────────

# A small MLP as the ODE drift. Realistic per-stage activation retention cost
# (many intermediate tensors per vf_prod eval) — matters for reverse-mode
# memory comparisons, where a cheap `y**3` VF would be dominated by solver-
# independent scratch and wouldn't reveal stage-count scaling.
_BENCH_DIM = 32
_BENCH_HIDDEN = 16
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
    t0=0.0,
    t1=1.0,
    tol=1e-3,
    shape=(_BENCH_DIM,),
    key=jax.random.key(0),
    levy_area=diffrax.SpaceTimeLevyArea,
)


def _bench_sde_diffusion(t, y, args):
    del t, args
    return lx.DiagonalLinearOperator(0.1 * y)


_BENCH_EUCLIDEAN_ODE_TERM = GeometricTerm(_bench_vf_global, geometry=Euclidean())
_BENCH_EUCLIDEAN_ODE_TERM_ARGS = GeometricTerm(_bench_vf_args, geometry=Euclidean())
_BENCH_EUCLIDEAN_SDE_TERM = MultiTerm(
    GeometricTerm(_bench_vf_global, geometry=Euclidean()),
    ControlTerm(_bench_sde_diffusion, _bench_bm_euclidean),
)
_BENCH_EUCLIDEAN_SDE_TERM_ARGS = MultiTerm(
    GeometricTerm(_bench_vf_args, geometry=Euclidean()),
    ControlTerm(_bench_sde_diffusion, _bench_bm_euclidean),
)

# ── SO(N) MLP VF ──────────────────────────────────────────────────────────────

# Use a high-dimensional SO(N) benchmark so the manifold state and Lie algebra
# coordinates are large enough to expose solver-owned stage storage costs.
_SON_N = 32
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

_SON_GEOMETRY = SO(_SON_N)


def _son_apply(weights, R):
    w_in, w_mid_layers, w_out = weights
    h = jnp.tanh(w_in @ R.flatten())
    for w_mid in w_mid_layers:
        h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)  # (dim so(n),) frame coords


# For the SDE, noise lives in the Lie algebra: dW in R^dim maps directly to
# frame coordinates via constant diagonal scaling.
_SON_NOISE_SCALE = jnp.full((_SON_DIM,), 0.05)
_bench_bm_son = diffrax.VirtualBrownianTree(
    t0=0.0,
    t1=1.0,
    tol=1e-3,
    shape=(_SON_DIM,),
    key=jax.random.key(1),
    levy_area=diffrax.SpaceTimeLevyArea,
)


def _son_sde_diffusion(t, R, args):
    del t, R, args
    return lx.DiagonalLinearOperator(_SON_NOISE_SCALE)


_BENCH_SON_ODE_TERM_GLOBAL = GeometricTerm(
    lambda t, R, args: _son_apply(_SON_PARAMS, R),
    geometry=_SON_GEOMETRY,
)
_BENCH_SON_ODE_TERM_ARGS = GeometricTerm(
    lambda t, R, args: _son_apply(args, R),
    geometry=_SON_GEOMETRY,
)
_BENCH_SON_SDE_TERM_GLOBAL = MultiTerm(
    GeometricTerm(
        lambda t, R, args: _son_apply(_SON_PARAMS, R), geometry=_SON_GEOMETRY
    ),
    ControlTerm(_son_sde_diffusion, _bench_bm_son),
)
_BENCH_SON_SDE_TERM_ARGS = MultiTerm(
    GeometricTerm(lambda t, R, args: _son_apply(args, R), geometry=_SON_GEOMETRY),
    ControlTerm(_son_sde_diffusion, _bench_bm_son),
)

# ── SPD(N) MLP VF ─────────────────────────────────────────────────────────────

_SPD_N = 6
_SPD_DIM = _SPD_N * (_SPD_N + 1) // 2  # 528, symmetric frame coords
_SPD_HIDDEN = 32
_SPD_AMBIENT_DIM = _SPD_N * _SPD_N
_SPD_NUM_LAYERS = 3
_spd_keys = jax.random.split(jax.random.key(13), _SPD_NUM_LAYERS)
_SPD_W_IN = jax.random.normal(_spd_keys[0], (_SPD_HIDDEN, _SPD_AMBIENT_DIM)) / jnp.sqrt(
    float(_SPD_AMBIENT_DIM)
)
_SPD_W_MID = tuple(
    jax.random.normal(key, (_SPD_HIDDEN, _SPD_HIDDEN)) / jnp.sqrt(float(_SPD_HIDDEN))
    for key in _spd_keys[1:-1]
)
_SPD_W_OUT = jax.random.normal(_spd_keys[-1], (_SPD_DIM, _SPD_HIDDEN)) / jnp.sqrt(
    float(_SPD_HIDDEN)
)
_SPD_PARAMS = (_SPD_W_IN, _SPD_W_MID, _SPD_W_OUT)

_SPD_GEOMETRY = SPD(_SPD_N)


def _spd_apply(weights, S):
    w_in, w_mid_layers, w_out = weights
    h = jnp.tanh(w_in @ S.flatten())
    for w_mid in w_mid_layers:
        h = jnp.tanh(w_mid @ h)
    return -0.1 * (w_out @ h)  # (dim spd(n),) frame coords


_SPD_NOISE_SCALE = jnp.full((_SPD_DIM,), 0.05)
_bench_bm_spd = diffrax.VirtualBrownianTree(
    t0=0.0,
    t1=1.0,
    tol=1e-3,
    shape=(_SPD_DIM,),
    key=jax.random.key(2),
    levy_area=diffrax.SpaceTimeLevyArea,
)


def _spd_sde_diffusion(t, S, args):
    del t, S, args
    return lx.DiagonalLinearOperator(_SPD_NOISE_SCALE)


_BENCH_SPD_ODE_TERM_GLOBAL = GeometricTerm(
    lambda t, S, args: _spd_apply(_SPD_PARAMS, S),
    geometry=_SPD_GEOMETRY,
)
_BENCH_SPD_ODE_TERM_ARGS = GeometricTerm(
    lambda t, S, args: _spd_apply(args, S),
    geometry=_SPD_GEOMETRY,
)
_BENCH_SPD_SDE_TERM_GLOBAL = MultiTerm(
    GeometricTerm(
        lambda t, S, args: _spd_apply(_SPD_PARAMS, S), geometry=_SPD_GEOMETRY
    ),
    ControlTerm(_spd_sde_diffusion, _bench_bm_spd),
)
_BENCH_SPD_SDE_TERM_ARGS = MultiTerm(
    GeometricTerm(lambda t, S, args: _spd_apply(args, S), geometry=_SPD_GEOMETRY),
    ControlTerm(_spd_sde_diffusion, _bench_bm_spd),
)

# ── Benchmark Cases ────────────────────────────────────────────────────────────


def _bench_group(name, y0, ode_global, ode_args, sde_global, sde_args, params):
    # `global` keeps MLP weights baked into HLO, differentiating wrt `y0`.
    # `args` threads weights through args so reverse-mode must retain the full
    # activation path needed for parameter gradients.
    return [
        BenchCase(f"ode/{name}/global", ode_global, y0, None, "y0"),
        BenchCase(f"ode/{name}/args", ode_args, y0, params, "args"),
        BenchCase(f"sde/{name}/global", sde_global, y0, None, "y0"),
        BenchCase(f"sde/{name}/args", sde_args, y0, params, "args"),
    ]


BENCH_CASES = [
    *_bench_group(
        "euclidean",
        jnp.ones((_BENCH_DIM,), dtype=jnp.float32),
        _BENCH_EUCLIDEAN_ODE_TERM,
        _BENCH_EUCLIDEAN_ODE_TERM_ARGS,
        _BENCH_EUCLIDEAN_SDE_TERM,
        _BENCH_EUCLIDEAN_SDE_TERM_ARGS,
        _BENCH_PARAMS,
    ),
    *_bench_group(
        f"so{_SON_N}",
        jnp.eye(_SON_N, dtype=jnp.float32),
        _BENCH_SON_ODE_TERM_GLOBAL,
        _BENCH_SON_ODE_TERM_ARGS,
        _BENCH_SON_SDE_TERM_GLOBAL,
        _BENCH_SON_SDE_TERM_ARGS,
        _SON_PARAMS,
    ),
    *_bench_group(
        f"spd{_SPD_N}",
        jnp.eye(_SPD_N, dtype=jnp.float32),
        _BENCH_SPD_ODE_TERM_GLOBAL,
        _BENCH_SPD_ODE_TERM_ARGS,
        _BENCH_SPD_SDE_TERM_GLOBAL,
        _BENCH_SPD_SDE_TERM_ARGS,
        _SPD_PARAMS,
    ),
]
