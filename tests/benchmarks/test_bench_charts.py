from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from georax._geometry._charts import SOChart
from georax._geometry.spd import SPD
from georax._geometry.special_orthogonal import SO

BENCH_RUNTIME_ROUNDS = 40
BENCH_WARMUP_ROUNDS = 5
_KEY = jax.random.PRNGKey(42)


def _spd_state(n: int) -> jax.Array:
    raw = jax.random.normal(_KEY, (n, n))
    return raw @ raw.T + n * jnp.eye(n)


# ── SOChart benchmarks ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "order,n",
    [
        pytest.param(order, n, id=f"ord{order}-n{n}")
        for order in (2, 4, 8, 12)
        for n in (4, 8, 16, 32)
    ],
)
@pytest.mark.benchmark(group="charts/so")
def test_so_chart_apply(benchmark: Any, order: int, n: int) -> None:
    geo = SO(n)
    chart = SOChart(order)
    x = jnp.eye(n)
    a = jnp.ones(n * (n - 1) // 2) * 0.1
    apply_jit = jax.jit(chart.apply)
    jax.block_until_ready(apply_jit(x, a, geo))
    benchmark.pedantic(
        lambda: jax.block_until_ready(apply_jit(x, a, geo)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )


# ── frame-bracket benchmarks ──────────────────────────────────────────────────


@pytest.mark.benchmark(group="charts/so3/frame-bracket")
def test_so3_frame_bracket(benchmark: Any) -> None:
    geo = SO(3)
    x = jnp.eye(3)
    a = jnp.array([0.2, -0.1, 0.15])
    b = jnp.array([0.3, -0.4, 0.2])
    bracket_jit = eqx.filter_jit(geo.frame_bracket)
    jax.block_until_ready(bracket_jit(x, a, b))

    benchmark.pedantic(
        lambda: jax.block_until_ready(bracket_jit(x, a, b)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )


@pytest.mark.parametrize(
    "n",
    [pytest.param(n, id=f"n{n}") for n in (2, 4, 8, 16, 32)],
)
@pytest.mark.benchmark(group="charts/spd/frame-bracket")
def test_spd_frame_bracket(benchmark: Any, n: int) -> None:
    geo = SPD(n)
    x = _spd_state(n)
    d = geo.coordinate_shape[0]
    a = jnp.linspace(-0.2, 0.2, d)
    b = jnp.linspace(0.1, -0.15, d)
    bracket_jit = eqx.filter_jit(geo.frame_bracket)
    jax.block_until_ready(bracket_jit(x, a, b))

    benchmark.pedantic(
        lambda: jax.block_until_ready(bracket_jit(x, a, b)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )
