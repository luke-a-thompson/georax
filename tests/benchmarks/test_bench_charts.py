from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from georax._geometry._charts import SOChart
from georax._geometry.special_orthogonal import SO

BENCH_RUNTIME_ROUNDS = 40
BENCH_WARMUP_ROUNDS = 5
_KEY = jax.random.PRNGKey(42)


# ── SOChart benchmarks ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "order,n",
    [
        pytest.param(2,  4,  id="ord2-n4",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(2,  8,  id="ord2-n8",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(2,  16, id="ord2-n16", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(2,  32, id="ord2-n32", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(4,  4,  id="ord4-n4",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(4,  8,  id="ord4-n8",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(4,  16, id="ord4-n16", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(4,  32, id="ord4-n32", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(8,  4,  id="ord8-n4",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(8,  8,  id="ord8-n8",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(8,  16, id="ord8-n16", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(8,  32, id="ord8-n32", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(12, 4,  id="ord12-n4",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(12, 8,  id="ord12-n8",  marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(12, 16, id="ord12-n16", marks=pytest.mark.benchmark(group="charts/so")),
        pytest.param(12, 32, id="ord12-n32", marks=pytest.mark.benchmark(group="charts/so")),
    ],
)
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
