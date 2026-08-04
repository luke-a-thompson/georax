from __future__ import annotations

from typing import Any

import jax
import pytest

from georax._geometry._charts import (
    _cayley,
    _bbc_expm_4,
    _bbc_expm_8,
    _ps_expm_12,
    _quadratic_expm,
    _taylor_expm,
)

BENCH_RUNTIME_ROUNDS = 40
BENCH_WARMUP_ROUNDS = 5
_KEY = jax.random.PRNGKey(42)


def _skew(n: int) -> jax.Array:
    raw = jax.random.normal(_KEY, (n, n))
    return 0.1 * (raw - raw.T)


# ── expm scheme benchmarks ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,n",
    [
        pytest.param(
            fn,
            n,
            id=f"{name}-n{n}",
            marks=pytest.mark.benchmark(group=f"expm/{name}"),
        )
        for name, fn in (
            ("quadratic", _quadratic_expm),
            ("bbc4", _bbc_expm_4),
            ("bbc8", _bbc_expm_8),
            ("ps12", _ps_expm_12),
        )
        for n in (4, 8, 16, 32)
    ],
)
def test_expm_scheme(benchmark: Any, fn: Any, n: int) -> None:
    a = _skew(n)
    fn_jit = jax.jit(fn)
    jax.block_until_ready(fn_jit(a))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn_jit(a)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )


@pytest.mark.parametrize(
    "degree,n",
    [
        pytest.param(degree, n, id=f"deg{degree}-n{n}")
        for degree in (2, 4, 8, 12)
        for n in (4, 16, 32)
    ],
)
@pytest.mark.benchmark(group="expm/taylor-dispatch")
def test_taylor_expm_dispatch(benchmark: Any, degree: int, n: int) -> None:
    a = _skew(n)
    fn_jit = jax.jit(_taylor_expm, static_argnums=(1,))
    jax.block_until_ready(fn_jit(a, degree))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn_jit(a, degree)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )


# ── _cayley benchmarks ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n",
    [pytest.param(n, id=f"n{n}") for n in (4, 8, 16, 32)],
)
@pytest.mark.benchmark(group="expm/cayley")
def test_cayley(benchmark: Any, n: int) -> None:
    a = _skew(n)
    fn_jit = jax.jit(_cayley)
    jax.block_until_ready(fn_jit(a))
    benchmark.pedantic(
        lambda: jax.block_until_ready(fn_jit(a)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )
