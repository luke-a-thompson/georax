"""Pathwise stochastic convergence experiment for the georax CF solvers."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import diffrax
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
import numpy as np
from diffrax import ReversibleAdjoint
from diffrax._custom_types import Args, RealScalarLike
from fbm import FBM
from jaxtyping import Array

from georax import CFEES25, CFEES27, SO, GeometricTerm

matplotlib.use("Agg")

T0 = 0.0
T1 = 1.0
SIGMA = 1.0
SO3_X0 = jnp.eye(3, dtype=jnp.float64)
SO3_GEOMETRY = SO(3)

XLABEL = r"$\log_{10}(h)$"
YLABEL_FORWARD = r"$\log_{10}(\mathcal{E}(h))$"
YLABEL_BACKWARD = r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"


def so3_coeffs_matrix(t: RealScalarLike, x: Array, args: Args) -> Array:
    del t, args
    old_coeffs = SIGMA * jnp.array(
        [
            [0.9 + 0.2 * x[0, 0], 0.15 + 0.25 * x[0, 1]],
            [0.25 + 0.2 * x[1, 2], -0.35 + 0.2 * x[1, 1]],
            [0.1 + 0.3 * x[2, 0], 0.8 + 0.15 * x[2, 2]],
        ],
        dtype=x.dtype,
    )
    # Map legacy local coordinates [B1, B2, B3] to georax SO(3) basis:
    # [B1, B2, B3] = [-K12, K02, -K01].
    coeffs = jnp.stack(
        (-old_coeffs[2, :], old_coeffs[1, :], -old_coeffs[0, :]),
        axis=0,
    )
    return coeffs


def get_2d_bm(num_steps: int, length: float, key: jax.Array) -> np.ndarray:
    dt = length / num_steps
    dw = np.asarray(
        jax.random.normal(key, (num_steps, 2), dtype=jnp.float64) * np.sqrt(dt)
    )
    x = np.zeros((num_steps + 1, 2), dtype=np.float64)
    x[1:] = np.cumsum(dw, axis=0)
    return x


def get_2d_fbm(num_steps: int, hurst: float, length: float) -> np.ndarray:
    fbm = FBM(n=num_steps, hurst=hurst, length=length, method="daviesharte")
    return np.stack([fbm.fbm(), fbm.fbm()], axis=-1)


def make_matrix_method_runner(
    solver,
    *,
    geometry,
    coeffs_matrix_fn: Callable[[RealScalarLike, Array, Args], Array],
    y0_default: np.ndarray,
) -> Callable[..., np.ndarray]:
    stepsize_controller = diffrax.ConstantStepSize()
    adjoint = ReversibleAdjoint()

    @lru_cache(maxsize=None)
    def _compiled_runner(num_points: int):
        if num_points < 2:
            raise ValueError("Need at least two control points.")

        ts = jnp.linspace(T0, T1, num_points, dtype=jnp.float64)
        dt0 = float((T1 - T0) / (num_points - 1))
        saveat = diffrax.SaveAt(ts=ts)

        @jax.jit
        def _run(x: jax.Array, y0: jax.Array) -> jax.Array:
            control = diffrax.LinearInterpolation(ts=ts, ys=x)
            term = GeometricTerm(
                geometry=geometry,
                coeffs_prod=lambda t, y, args, dW: coeffs_matrix_fn(t, y, args) @ dW,
                control_fn=lambda t0, t1, **kwargs: control.evaluate(
                    t0, t1, **kwargs
                ),
            )
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=T0,
                t1=T1,
                dt0=dt0,
                y0=y0,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=num_points + 4,
                adjoint=adjoint,
                throw=True,
            )
            assert solution.ys is not None
            return solution.ys

        return _run

    def _run(x: np.ndarray, y0: np.ndarray | None = None) -> np.ndarray:
        x_arr = jnp.asarray(np.asarray(x, dtype=np.float64), dtype=jnp.float64)
        runner = _compiled_runner(int(x_arr.shape[0]))
        if y0 is None:
            y0 = np.asarray(y0_default, dtype=np.float64)
        y = runner(x_arr, jnp.asarray(y0, dtype=jnp.float64))
        return np.asarray(y, dtype=np.float64)

    return _run


def make_so3_method_runner(solver) -> Callable[..., np.ndarray]:
    return make_matrix_method_runner(
        solver,
        geometry=SO3_GEOMETRY,
        coeffs_matrix_fn=so3_coeffs_matrix,
        y0_default=np.asarray(SO3_X0, dtype=np.float64),
    )


def get_error(y_exact: np.ndarray, y_vals: np.ndarray, step: int) -> float:
    true_vals = y_exact[::step]
    return float(np.max(np.linalg.norm(true_vals - y_vals, axis=(1, 2))))


def matrix_distance(y0: np.ndarray, y1: np.ndarray) -> float:
    return float(np.linalg.norm(y0 - y1))


def compute_error_curves(
    method_run,
    paths: list[np.ndarray],
    hs: np.ndarray,
    *,
    error_fn: Callable[[np.ndarray, np.ndarray, int], float],
    distance_fn: Callable[[np.ndarray | float, np.ndarray | float], float],
) -> tuple[np.ndarray, np.ndarray]:
    forward_y = np.zeros(len(hs), dtype=np.float64)
    backward_y = np.zeros(len(hs), dtype=np.float64)

    for x in paths:
        n = len(x) - 1
        y_exact = method_run(x)

        forward_error = []
        backward_error = []
        for h in hs:
            step = max(1, int(round(n * h)))
            x_coarse = x[::step]
            y_coarse = method_run(x_coarse)
            forward_error.append(error_fn(y_exact, y_coarse, step))

            y_backward = method_run(x_coarse[::-1], y0=y_coarse[-1])
            backward_error.append(distance_fn(y_exact[0], y_backward[-1]))

        forward_y += np.log10(np.maximum(forward_error, np.finfo(np.float64).tiny))
        backward_y += np.log10(np.maximum(backward_error, np.finfo(np.float64).tiny))

    return forward_y / len(paths), backward_y / len(paths)


def plot_curve(
    name: str,
    h: np.ndarray,
    y: np.ndarray,
    slope: float,
    ax,
    *,
    backward: bool = False,
) -> None:
    x = np.log10(h)
    dx = np.array([x[0], x[-1]], dtype=np.float64)
    intercept = float(np.mean(y) - slope * np.mean(x))
    fit = np.polyfit(x, y, 1)
    err_label = YLABEL_BACKWARD if backward else YLABEL_FORWARD
    mode = "backward" if backward else "forward"
    print(f"{name} {mode} slope: {fit[0]:.6f}")

    ax.scatter(
        x,
        y,
        marker=mkr.MarkerStyle("x", fillstyle="none"),
        color="crimson",
    )
    ax.plot(dx, slope * dx + intercept, color="mediumblue")
    ax.legend([err_label, f"{np.round(slope, 1)}$x + c$"])
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(err_label)


def expected_rates(
    solver,
    hurst: float,
) -> tuple[float, float]:
    _get = lambda name: getattr(solver, name, lambda _: None)(None)
    order = _get("order")
    antisymmetric_order = _get("antisymmetric_order")
    if antisymmetric_order is None:
        if order is None:
            raise ValueError(f"Solver {solver} does not define an order.")
        antisymmetric_order = order + 1
    return 2.0 * hurst - 0.5, (antisymmetric_order + 1.0) * hurst - 1.0


def make_paths(
    num_paths: int, path_power: int, key: jax.Array, hurst: float = 0.5
) -> list[np.ndarray]:
    num_steps = 2**path_power
    if hurst == 0.5:
        keys = jax.random.split(key, num_paths)
        return [get_2d_bm(num_steps, T1 - T0, subkey) for subkey in keys]
    return [get_2d_fbm(num_steps, hurst, T1 - T0) for _ in range(num_paths)]


def plot_grid(
    paths: list[np.ndarray],
    hs: np.ndarray,
    hurst: float,
    output_dir: Path,
) -> Path:
    benchmarks = [
        (
            "CF-EES25",
            CFEES25(),
            make_so3_method_runner,
            get_error,
            matrix_distance,
            r"$\mathcal{E}(h)$ for $\mathrm{CF\text{-}EES}_\mathcal{R}(2,5)$",
            r"$\overleftarrow{\mathcal{E}}(h)$ for $\mathrm{CF\text{-}EES}_\mathcal{R}(2,5)$",
        ),
        (
            "CF-EES27",
            CFEES27(),
            make_so3_method_runner,
            get_error,
            matrix_distance,
            r"$\mathcal{E}(h)$ for $\mathrm{CF\text{-}EES}_\mathcal{R}(2,7)$",
            r"$\overleftarrow{\mathcal{E}}(h)$ for $\mathrm{CF\text{-}EES}_\mathcal{R}(2,7)$",
        ),
    ]

    fig, axes = plt.subplots(len(benchmarks), 2, figsize=(10, 3.2 * len(benchmarks)))

    for i, (
        name,
        solver,
        runner_factory,
        error_fn,
        distance_fn,
        fwd_title,
        bwd_title,
    ) in enumerate(benchmarks):
        runner = runner_factory(solver)
        forward_y, backward_y = compute_error_curves(
            runner,
            paths,
            hs,
            error_fn=error_fn,
            distance_fn=distance_fn,
        )
        forward_rate, backward_rate = expected_rates(solver, hurst)

        plot_curve(name, hs, forward_y, forward_rate, axes[i][0], backward=False)
        axes[i][0].set_title(fwd_title)

        plot_curve(name, hs, backward_y, backward_rate, axes[i][1], backward=True)
        axes[i][1].set_title(bwd_title)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stochastic_convergence_H{int(hurst * 100)}.pdf"
    plt.savefig(output_path)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=12)
    parser.add_argument("--path-power", type=int, default=13)
    parser.add_argument("--hurst", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--min-power", type=int, default=3)
    parser.add_argument("--max-power", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hs = np.array(
        [2.0 ** (-k) for k in range(args.min_power, args.max_power + 1)],
        dtype=np.float64,
    )
    key = jax.random.key(args.seed)
    for hurst in args.hurst:
        np.random.seed(args.seed)
        paths = make_paths(args.num_paths, args.path_power, key, hurst=hurst)
        output_path = plot_grid(paths, hs, hurst, args.output_dir)
        print(f"H={hurst:.1f} saved {output_path}")


if __name__ == "__main__":
    main()
