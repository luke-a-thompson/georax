from __future__ import annotations

import time

import diffrax
import jax
import jax.numpy as jnp
import pytest
from conftest import BENCH_SOLVERS, EuclideanOps

from georax import GeometricTerm

def _reverse_mode_adjoints_for_solver(solver_cls):
    adjoints = [("recursive_checkpoint", diffrax.RecursiveCheckpointAdjoint())]
    if issubclass(solver_cls, diffrax.AbstractReversibleSolver):
        adjoints.append(("reversible", diffrax.ReversibleAdjoint()))
    return adjoints


def _to_total_bytes(memory_stats) -> int:
    return int(
        memory_stats.temp_size_in_bytes
        + memory_stats.argument_size_in_bytes
        + memory_stats.output_size_in_bytes
        - memory_stats.alias_size_in_bytes
    )


def _get_memory_stats(compiled):
    if not hasattr(compiled, "memory_analysis"):
        pytest.skip("Compiled executable does not expose memory_analysis().")
    memory_stats = compiled.memory_analysis()
    if memory_stats is None:
        pytest.skip("memory_analysis() returned None on this backend.")
    return memory_stats


def _make_term(vf) -> GeometricTerm:
    return GeometricTerm(inner=diffrax.ODETerm(vf), geometry=EuclideanOps())


def _compiled_memory_bytes(solver_cls, y0):
    solver = solver_cls()
    term = _make_term(lambda t, y, args: -10.0 * y**3)
    saveat = diffrax.SaveAt(t1=True)

    def run(y_init):
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=y_init,
            saveat=saveat,
            throw=True,
        )
        return out.ys

    compiled = jax.jit(run).lower(y0).compile()
    memory_stats = _get_memory_stats(compiled)
    return _to_total_bytes(memory_stats), memory_stats


def _compiled_reverse_mode_memory_bytes(
    solver_cls, y0, *, adjoint=None, max_steps=None
):
    solver = solver_cls()
    term = _make_term(lambda t, y, args: -10.0 * y**3)
    saveat = diffrax.SaveAt(t1=True)
    solve_kwargs = {}
    if adjoint is not None:
        solve_kwargs["adjoint"] = adjoint
    if max_steps is not None:
        solve_kwargs["max_steps"] = max_steps

    def loss(y_init):
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=y_init,
            saveat=saveat,
            throw=True,
            **solve_kwargs,
        )
        assert out.ys is not None
        return jnp.sum(out.ys)

    compiled = jax.jit(jax.grad(loss)).lower(y0).compile()
    memory_stats = _get_memory_stats(compiled)
    return _to_total_bytes(memory_stats), memory_stats


@pytest.mark.parametrize("problem_size", [8192])
def test_solvers_compiled_memory(problem_size):
    y0 = jnp.ones((problem_size,), dtype=jnp.float32)

    results = []
    for solver_name, solver_cls in BENCH_SOLVERS:
        total, _ = _compiled_memory_bytes(solver_cls, y0)
        results.append((solver_name, total))

    print(f"\ncompiled-memory-bytes (size={problem_size}):")
    for name, total in results:
        ratio = total / results[0][1] if results[0][1] else float("inf")
        print(f"  {name}: {total} bytes  (vs {results[0][0]}: {ratio:.3f}x)")
        assert total > 0, f"{name} reported non-positive compiled-memory bytes."


@pytest.mark.parametrize("problem_size", [8192])
def test_solvers_compiled_reverse_mode_memory_by_adjoint(problem_size):
    y0 = jnp.ones((problem_size,), dtype=jnp.float32)
    max_steps = 4096

    print(f"\ncompiled-reverse-mode-memory-bytes-by-adjoint (size={problem_size}):")
    for solver_name, solver_cls in BENCH_SOLVERS:
        print(f"  solver={solver_name}:")
        for adjoint_name, adjoint in _reverse_mode_adjoints_for_solver(solver_cls):
            total, _ = _compiled_reverse_mode_memory_bytes(
                solver_cls,
                y0,
                adjoint=adjoint,
                max_steps=max_steps,
            )
            print(f"    adjoint={adjoint_name}: {total} bytes")
            assert total > 0, (
                f"{solver_name} with adjoint={adjoint_name} reported non-positive "
                "reverse-mode compiled-memory bytes."
            )


def _runtime_seconds(solver_cls, y0, n_repeats=100):
    solver = solver_cls()
    term = _make_term(lambda t, y, args: -10.0 * y**3)
    saveat = diffrax.SaveAt(t1=True)

    @jax.jit
    def run(y_init):
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=y_init,
            saveat=saveat,
            throw=True,
        )
        return out.ys

    # Warmup
    jax.block_until_ready(run(y0))

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(run(y0))
    return (time.perf_counter() - t0) / n_repeats


@pytest.mark.parametrize("problem_size", [8192])
def test_solvers_runtime(problem_size):
    y0 = jnp.ones((problem_size,), dtype=jnp.float32)

    results = []
    for solver_name, solver_cls in BENCH_SOLVERS:
        t = _runtime_seconds(solver_cls, y0)
        results.append((solver_name, t))

    print(f"\nruntime (size={problem_size}):")
    for name, t in results:
        ratio = t / results[0][1] if results[0][1] else float("inf")
        print(f"  {name}: {t * 1e3:.3f} ms  (vs {results[0][0]}: {ratio:.3f}x)")
        assert t > 0, f"{name} reported non-positive runtime."
