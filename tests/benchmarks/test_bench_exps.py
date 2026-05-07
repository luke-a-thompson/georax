from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
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
        pytest.param(_quadratic_expm, 4,  id="quadratic-n4",  marks=pytest.mark.benchmark(group="expm/quadratic")),
        pytest.param(_quadratic_expm, 8,  id="quadratic-n8",  marks=pytest.mark.benchmark(group="expm/quadratic")),
        pytest.param(_quadratic_expm, 16, id="quadratic-n16", marks=pytest.mark.benchmark(group="expm/quadratic")),
        pytest.param(_quadratic_expm, 32, id="quadratic-n32", marks=pytest.mark.benchmark(group="expm/quadratic")),
        pytest.param(_bbc_expm_4,     4,  id="bbc4-n4",       marks=pytest.mark.benchmark(group="expm/bbc4")),
        pytest.param(_bbc_expm_4,     8,  id="bbc4-n8",       marks=pytest.mark.benchmark(group="expm/bbc4")),
        pytest.param(_bbc_expm_4,     16, id="bbc4-n16",      marks=pytest.mark.benchmark(group="expm/bbc4")),
        pytest.param(_bbc_expm_4,     32, id="bbc4-n32",      marks=pytest.mark.benchmark(group="expm/bbc4")),
        pytest.param(_bbc_expm_8,     4,  id="bbc8-n4",       marks=pytest.mark.benchmark(group="expm/bbc8")),
        pytest.param(_bbc_expm_8,     8,  id="bbc8-n8",       marks=pytest.mark.benchmark(group="expm/bbc8")),
        pytest.param(_bbc_expm_8,     16, id="bbc8-n16",      marks=pytest.mark.benchmark(group="expm/bbc8")),
        pytest.param(_bbc_expm_8,     32, id="bbc8-n32",      marks=pytest.mark.benchmark(group="expm/bbc8")),
        pytest.param(_ps_expm_12,     4,  id="ps12-n4",       marks=pytest.mark.benchmark(group="expm/ps12")),
        pytest.param(_ps_expm_12,     8,  id="ps12-n8",       marks=pytest.mark.benchmark(group="expm/ps12")),
        pytest.param(_ps_expm_12,     16, id="ps12-n16",      marks=pytest.mark.benchmark(group="expm/ps12")),
        pytest.param(_ps_expm_12,     32, id="ps12-n32",      marks=pytest.mark.benchmark(group="expm/ps12")),
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
        pytest.param(2,  4,  id="deg2-n4",   marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(2,  16, id="deg2-n16",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(2,  32, id="deg2-n32",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(4,  4,  id="deg4-n4",   marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(4,  16, id="deg4-n16",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(4,  32, id="deg4-n32",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(8,  4,  id="deg8-n4",   marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(8,  16, id="deg8-n16",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(8,  32, id="deg8-n32",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(12, 4,  id="deg12-n4",  marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(12, 16, id="deg12-n16", marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
        pytest.param(12, 32, id="deg12-n32", marks=pytest.mark.benchmark(group="expm/taylor-dispatch")),
    ],
)
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
    [
        pytest.param(4,  id="n4",  marks=pytest.mark.benchmark(group="expm/cayley")),
        pytest.param(8,  id="n8",  marks=pytest.mark.benchmark(group="expm/cayley")),
        pytest.param(16, id="n16", marks=pytest.mark.benchmark(group="expm/cayley")),
        pytest.param(32, id="n32", marks=pytest.mark.benchmark(group="expm/cayley")),
    ],
)
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
