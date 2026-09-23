"""Plot estimated adjoint memory scaling versus timestep count for benchmark states."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _benchmarks import BENCH_CASES, BENCH_SOLVERS

from georax import SO, Euclidean

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_NUM_STEPS = (10, 25, 50, 200, 400, 600, 800)
DEFAULT_X_TICKS = (0, 200, 400, 600, 800)


def state_equivalent_bytes(y0) -> int:
    return int(np.asarray(y0).nbytes)


def solver_memory_scaling_bytes(
    *, solver, y0, num_steps_values: np.ndarray
) -> tuple[np.ndarray, str]:
    # This is a state-equivalent memory model, not an empirical RSS measurement.
    # `DirectAdjoint` retains per-step primal state, while `ReversibleAdjoint`
    # only keeps O(1) solver state and reconstructs the trajectory backward.
    state_bytes = float(state_equivalent_bytes(y0))
    if isinstance(solver, diffrax.AbstractReversibleSolver):
        return (
            np.full_like(num_steps_values, fill_value=state_bytes, dtype=np.float64),
            "reversible",
        )
    if not hasattr(solver, "tableau"):
        raise TypeError(f"Unsupported solver type for memory model: {type(solver)!r}")
    stage_count = len(solver.tableau.c)
    return state_bytes * stage_count * num_steps_values.astype(np.float64), "direct"


def collect_memory_curve(
    *,
    y0,
    solver_entries,
    num_steps_values: np.ndarray,
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for solver_name, solver_cls in solver_entries:
        values, adjoint_name = solver_memory_scaling_bytes(
            solver=solver_cls(),
            y0=y0,
            num_steps_values=num_steps_values,
        )
        curves[f"{solver_name} ({adjoint_name})"] = values
    return curves


def choose_memory_unit(max_bytes: float) -> tuple[float, str]:
    if max_bytes >= 1e9:
        return 1e9, "GB"
    if max_bytes >= 1e6:
        return 1e6, "MB"
    if max_bytes >= 1e3:
        return 1e3, "KB"
    return 1.0, "B"


def plot_panel(
    ax,
    *,
    title: str,
    num_steps_values: np.ndarray,
    curves: dict[str, np.ndarray],
    unit_scale: float,
    unit_label: str,
) -> None:
    for solver_name, values in curves.items():
        ax.plot(
            num_steps_values,
            values / unit_scale,
            marker="o",
            linewidth=1.8,
            label=solver_name,
        )
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"Estimated retained state-equivalent memory ({unit_label})")
    if np.max(num_steps_values) <= DEFAULT_X_TICKS[-1]:
        ax.set_xticks(DEFAULT_X_TICKS)
    ax.set_xlim(0, max(DEFAULT_X_TICKS[-1], np.max(num_steps_values)))
    ax.set_yscale("log", base=2)
    ax.margins(y=0.15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def plot_grid(
    *,
    num_steps_values: np.ndarray,
    output_dir: Path,
) -> Path:
    solver_entries = BENCH_SOLVERS
    benchmark_specs = []
    for case in BENCH_CASES:
        if not (case.name.startswith("ode/") and case.grad_target == "y0"):
            continue
        geometry = case.term.geometry
        if isinstance(geometry, Euclidean):
            title = f"Euclidean ODE (dim {case.y0.size})"
        elif isinstance(geometry, SO):
            title = f"SO({geometry.n}) ODE"
        else:
            continue
        benchmark_specs.append((title, case.y0))
    all_curves: list[dict[str, np.ndarray]] = []
    max_bytes = 0.0
    for _, case_y0 in benchmark_specs:
        curves = collect_memory_curve(
            y0=case_y0,
            solver_entries=solver_entries,
            num_steps_values=num_steps_values,
        )
        all_curves.append(curves)
        max_bytes = max(max_bytes, *(np.max(values) for values in curves.values()))
    unit_scale, unit_label = choose_memory_unit(max_bytes)

    fig, axes = plt.subplots(1, len(benchmark_specs), figsize=(12, 4.8), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, (title, _), curves in zip(axes, benchmark_specs, all_curves, strict=True):
        plot_panel(
            ax,
            title=title,
            num_steps_values=num_steps_values,
            curves=curves,
            unit_scale=unit_scale,
            unit_label=unit_label,
        )

    fig.suptitle("Adjoint memory scaling: direct O(N), reversible O(1)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "memory_requirements.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_STEPS,
        help="Explicit timestep counts to plot on the x-axis.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_steps_values = np.asarray(args.num_steps, dtype=np.int32)
    if np.any(num_steps_values <= 0):
        raise ValueError("--num-steps must be strictly positive.")
    output_path = plot_grid(
        num_steps_values=num_steps_values,
        output_dir=args.output_dir,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
