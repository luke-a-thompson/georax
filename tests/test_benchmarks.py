from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import diffrax
import jax
import jax.numpy as jnp
import pytest
from conftest import BENCH_CASES, BENCH_SOLVERS, BenchCase
from jaxtyping import PyTree

from georax import GeometricTerm

BENCH_NUM_STEPS = 512
BENCH_DT0 = (1.0 - 0.0) / BENCH_NUM_STEPS
BENCH_RUNTIME_REPEATS = 2

REVERSE_MODE_ADJOINTS = [
    ("recursive_checkpoint", diffrax.RecursiveCheckpointAdjoint()),
    # ("direct", diffrax.DirectAdjoint()),
    ("reversible", diffrax.ReversibleAdjoint()),
]


def _get_memory_stats(compiled: Any) -> Any:
    if not hasattr(compiled, "memory_analysis"):
        pytest.skip("Compiled executable does not expose memory_analysis().")
    memory_stats = compiled.memory_analysis()
    if memory_stats is None:
        pytest.skip("memory_analysis() returned None on this backend.")
    return memory_stats


def _compiled_memory_bytes(
    solver_cls: type[diffrax.AbstractSolver],
    y0: PyTree,
    term: GeometricTerm,
    args: PyTree | None,
) -> int:
    solver = solver_cls()
    saveat = diffrax.SaveAt(t1=True)

    def run(y_init: PyTree, solve_args: PyTree | None) -> PyTree:
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=BENCH_DT0,
            y0=y_init,
            args=solve_args,
            saveat=saveat,
            throw=True,
        )
        return out.ys

    compiled = jax.jit(run).lower(y0, args).compile()
    return int(_get_memory_stats(compiled).temp_size_in_bytes)


def _compiled_reverse_mode_memory_bytes(
    solver_cls: type[diffrax.AbstractSolver],
    y0: PyTree,
    term: GeometricTerm,
    args: PyTree | None,
    grad_target: Literal["y0", "args"],
    *,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int | None = None,
) -> int:
    solver = solver_cls()
    saveat = diffrax.SaveAt(t1=True)
    solve_kwargs: dict[str, Any] = {}
    if adjoint is not None:
        solve_kwargs["adjoint"] = adjoint
    if max_steps is not None:
        solve_kwargs["max_steps"] = max_steps

    def loss(y_init: PyTree, solve_args: PyTree | None) -> Any:
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=BENCH_DT0,
            y0=y_init,
            args=solve_args,
            saveat=saveat,
            throw=True,
            **solve_kwargs,
        )
        assert out.ys is not None
        return jnp.sum(out.ys)

    if grad_target == "y0":
        compiled = (
            jax.jit(jax.value_and_grad(lambda y_init: loss(y_init, args)))
            .lower(y0)
            .compile()
        )
    elif grad_target == "args":
        compiled = (
            jax.jit(jax.value_and_grad(lambda solve_args: loss(y0, solve_args)))
            .lower(args)
            .compile()
        )
    else:
        raise ValueError(f"Unsupported grad target: {grad_target}")

    return int(_get_memory_stats(compiled).temp_size_in_bytes)


def _runtime_seconds(
    solver_cls: type[diffrax.AbstractSolver],
    y0: PyTree,
    term: GeometricTerm,
    args: PyTree | None,
    n_repeats: int = 100,
) -> float:
    solver = solver_cls()
    saveat = diffrax.SaveAt(t1=True)

    @jax.jit
    def run(y_init: PyTree, solve_args: PyTree | None) -> PyTree:
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=BENCH_DT0,
            y0=y_init,
            args=solve_args,
            saveat=saveat,
            throw=True,
        )
        return out.ys

    jax.block_until_ready(run(y0, args))

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(run(y0, args))
    return (time.perf_counter() - t0) / n_repeats


def _grad_runtime_seconds(
    solver_cls: type[diffrax.AbstractSolver],
    y0: PyTree,
    term: GeometricTerm,
    args: PyTree | None,
    grad_target: Literal["y0", "args"],
    n_repeats: int = 100,
) -> float:
    solver = solver_cls()
    saveat = diffrax.SaveAt(t1=True)

    def loss(y_init: PyTree, solve_args: PyTree | None) -> Any:
        out = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=BENCH_DT0,
            y0=y_init,
            args=solve_args,
            saveat=saveat,
            throw=True,
        )
        assert out.ys is not None
        return jnp.sum(out.ys)

    if grad_target == "y0":
        run = jax.jit(jax.value_and_grad(lambda y_init: loss(y_init, args)))
        run_args = (y0,)
    elif grad_target == "args":
        run = jax.jit(jax.value_and_grad(lambda solve_args: loss(y0, solve_args)))
        run_args = (args,)
    else:
        raise ValueError(f"Unsupported grad target: {grad_target}")

    jax.block_until_ready(run(*run_args))

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(run(*run_args))
    return (time.perf_counter() - t0) / n_repeats


def _print_table(
    header: str,
    rows: Sequence[tuple[str, int | float]],
    value_fmt: Callable[[int | float], str],
) -> None:
    w = max(len(name) for name, _ in rows)
    print(f"\n{header}")
    base = rows[0][1]
    for name, val in rows:
        ratio = val / base if base else float("inf")
        print(f"  {name:<{w}}  {value_fmt(val)}  {ratio:.2f}x")


@pytest.mark.benchmark
@pytest.mark.parametrize("case", BENCH_CASES, ids=[c.name for c in BENCH_CASES])
def test_solvers_compiled_memory(case: BenchCase) -> None:
    results: list[tuple[str, int]] = []
    for solver_name, solver_cls in BENCH_SOLVERS:
        total = _compiled_memory_bytes(solver_cls, case.y0, case.term, case.args)
        results.append((solver_name, total))

    _print_table(
        f"compiled-memory [{case.name}]",
        results,
        lambda b: f"{b / 1e6:6.2f} MB",
    )
    for name, total in results:
        assert total > 0, f"{name} reported non-positive compiled-memory bytes."


@pytest.mark.benchmark
@pytest.mark.parametrize("case", BENCH_CASES, ids=[c.name for c in BENCH_CASES])
@pytest.mark.parametrize(
    "adjoint_name,adjoint",
    REVERSE_MODE_ADJOINTS,
    ids=[name for name, _ in REVERSE_MODE_ADJOINTS],
)
def test_solvers_compiled_reverse_mode_memory_by_adjoint(
    case: BenchCase,
    adjoint_name: str,
    adjoint: diffrax.AbstractAdjoint,
) -> None:
    max_steps = BENCH_NUM_STEPS * 2

    solvers = BENCH_SOLVERS
    if isinstance(adjoint, diffrax.ReversibleAdjoint):
        solvers = [
            (name, cls)
            for name, cls in BENCH_SOLVERS
            if issubclass(cls, diffrax.AbstractReversibleSolver)
        ]
        if not solvers:
            pytest.skip("No AbstractReversibleSolver in BENCH_SOLVERS.")

    results: list[tuple[str, int]] = []
    for solver_name, solver_cls in solvers:
        total = _compiled_reverse_mode_memory_bytes(
            solver_cls,
            case.y0,
            case.term,
            case.args,
            case.grad_target,
            adjoint=adjoint,
            max_steps=max_steps,
        )
        results.append((solver_name, total))

    _print_table(
        f"grad-memory [{adjoint_name}/{case.name}]",
        results,
        lambda b: f"{b / 1e6:6.2f} MB",
    )
    for name, total in results:
        assert total > 0, (
            f"{name} with adjoint={adjoint_name} reported non-positive "
            "reverse-mode compiled-memory bytes."
        )


@pytest.mark.benchmark
@pytest.mark.parametrize("case", BENCH_CASES, ids=[c.name for c in BENCH_CASES])
def test_solvers_runtime(case: BenchCase) -> None:
    results: list[tuple[str, float]] = []
    for solver_name, solver_cls in BENCH_SOLVERS:
        t = _runtime_seconds(
            solver_cls,
            case.y0,
            case.term,
            case.args,
            n_repeats=BENCH_RUNTIME_REPEATS,
        )
        results.append((solver_name, t))

    _print_table(
        f"runtime [{case.name}]",
        results,
        lambda t: f"{t * 1e3:7.2f} ms",
    )
    for name, t in results:
        assert t > 0, f"{name} reported non-positive runtime."


@pytest.mark.benchmark
@pytest.mark.parametrize("case", BENCH_CASES, ids=[c.name for c in BENCH_CASES])
def test_solvers_grad_runtime(case: BenchCase) -> None:
    results: list[tuple[str, float]] = []
    for solver_name, solver_cls in BENCH_SOLVERS:
        t = _grad_runtime_seconds(
            solver_cls,
            case.y0,
            case.term,
            case.args,
            case.grad_target,
            n_repeats=BENCH_RUNTIME_REPEATS,
        )
        results.append((solver_name, t))

    _print_table(
        f"grad-runtime [{case.name}]",
        results,
        lambda t: f"{t * 1e3:7.2f} ms",
    )
    for name, t in results:
        assert t > 0, f"{name} reported non-positive grad runtime."
