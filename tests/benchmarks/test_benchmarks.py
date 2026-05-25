from __future__ import annotations

from typing import Any, Literal

import diffrax
import jax
import jax.numpy as jnp
import pytest
from conftest import BENCH_CASES, BenchCase, bench_solvers_for_case
from jaxtyping import PyTree

from georax import GeometricTerm

BENCH_NUM_STEPS = 512
BENCH_DT0 = (1.0 - 0.0) / BENCH_NUM_STEPS
BENCH_RUNTIME_ROUNDS = 2

REVERSE_MODE_ADJOINTS = [
    ("recursive_checkpoint", diffrax.RecursiveCheckpointAdjoint()),
    ("reversible", diffrax.ReversibleAdjoint()),
]


# ── Parametrize helpers ───────────────────────────────────────────────────────


# Forward-only benches don't differentiate, so the `/global` cases (weights
# baked into HLO) add no signal over `/args`.
FORWARD_CASES = [c for c in BENCH_CASES if c.grad_target == "args"]


def _memory_params() -> list[Any]:
    out = []
    for case in FORWARD_CASES:
        for solver_name, solver_cls in bench_solvers_for_case(case):
            out.append(pytest.param(case, solver_cls, id=f"{case.name}-{solver_name}"))
    return out


def _reverse_memory_params() -> list[Any]:
    out = []
    for case in BENCH_CASES:
        for solver_name, solver_cls in bench_solvers_for_case(case):
            for adjoint_name, adjoint in REVERSE_MODE_ADJOINTS:
                out.append(
                    pytest.param(
                        case,
                        solver_cls,
                        adjoint_name,
                        adjoint,
                        id=f"{case.name}-{adjoint_name}-{solver_name}",
                    )
                )
    return out


def _runtime_params(cases: list[BenchCase], group_prefix: str) -> list[Any]:
    out = []
    for case in cases:
        for solver_name, solver_cls in bench_solvers_for_case(case):
            out.append(
                pytest.param(
                    case,
                    solver_cls,
                    id=f"{case.name}-{solver_name}",
                    marks=pytest.mark.benchmark(group=f"{group_prefix}/{case.name}"),
                )
            )
    return out


# ── Solver builders ───────────────────────────────────────────────────────────


def _make_forward(solver_cls: Any, term: GeometricTerm) -> Any:
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

    return run


def _make_grad(
    solver_cls: Any,
    y0: PyTree,
    term: GeometricTerm,
    args: PyTree | None,
    grad_target: Literal["y0", "args"],
    *,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int | None = None,
) -> tuple[Any, tuple[Any, ...]]:
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
        return jax.value_and_grad(lambda y_init: loss(y_init, args)), (y0,)
    if grad_target == "args":
        return jax.value_and_grad(lambda solve_args: loss(y0, solve_args)), (args,)
    raise ValueError(f"Unsupported grad target: {grad_target}")


def _temp_size_bytes(compiled: Any) -> int:
    if not hasattr(compiled, "memory_analysis"):
        pytest.skip("Compiled executable does not expose memory_analysis().")
    stats = compiled.memory_analysis()
    if stats is None:
        pytest.skip("memory_analysis() returned None on this backend.")
    return int(stats.temp_size_in_bytes)


# ── Memory benchmarks (record_property; opt-in via -m benchmark) ──────────────


@pytest.mark.benchmark
@pytest.mark.parametrize("case,solver_cls", _memory_params())
def test_compiled_memory(
    record_property: Any, case: BenchCase, solver_cls: Any
) -> None:
    run = _make_forward(solver_cls, case.term)
    compiled = jax.jit(run).lower(case.y0, case.args).compile()
    total = _temp_size_bytes(compiled)
    record_property("compiled_memory_bytes", total)
    assert total > 0


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "case,solver_cls,adjoint_name,adjoint", _reverse_memory_params()
)
def test_compiled_reverse_mode_memory(
    record_property: Any,
    case: BenchCase,
    solver_cls: Any,
    adjoint_name: str,
    adjoint: diffrax.AbstractAdjoint,
) -> None:
    if isinstance(adjoint, diffrax.ReversibleAdjoint) and not isinstance(
        solver_cls(), diffrax.AbstractReversibleSolver
    ):
        pytest.skip("Solver is not an AbstractReversibleSolver.")

    fn, run_args = _make_grad(
        solver_cls,
        case.y0,
        case.term,
        case.args,
        case.grad_target,
        adjoint=adjoint,
        max_steps=BENCH_NUM_STEPS * 2,
    )
    compiled = jax.jit(fn).lower(*run_args).compile()
    total = _temp_size_bytes(compiled)
    record_property("compiled_memory_bytes", total)
    record_property("adjoint", adjoint_name)
    assert total > 0


# ── Runtime benchmarks (pytest-benchmark) ─────────────────────────────────────


@pytest.mark.parametrize("case,solver_cls", _runtime_params(FORWARD_CASES, "runtime"))
def test_runtime(benchmark: Any, case: BenchCase, solver_cls: Any) -> None:
    run = jax.jit(_make_forward(solver_cls, case.term))
    jax.block_until_ready(run(case.y0, case.args))

    benchmark.pedantic(
        lambda: jax.block_until_ready(run(case.y0, case.args)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=1,
    )


@pytest.mark.parametrize(
    "case,solver_cls", _runtime_params(BENCH_CASES, "grad-runtime")
)
def test_grad_runtime(benchmark: Any, case: BenchCase, solver_cls: Any) -> None:
    fn, run_args = _make_grad(
        solver_cls, case.y0, case.term, case.args, case.grad_target
    )
    run = jax.jit(fn)
    jax.block_until_ready(run(*run_args))

    benchmark.pedantic(
        lambda: jax.block_until_ready(run(*run_args)),
        iterations=1,
        rounds=BENCH_RUNTIME_ROUNDS,
        warmup_rounds=1,
    )
