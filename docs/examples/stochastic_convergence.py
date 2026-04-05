"""Pathwise stochastic convergence experiment for the georax CF solvers."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diffrax
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
import numpy as np
from diffrax import AbstractReversibleSolver, ReversibleAdjoint
from diffrax_lowstorage import LowStorageRecurrence
from diffrax_lowstorage.base import LowStorageSolver
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array

from georax import CFEES25, CFEES27, GeometricTerm, LieGroup, SPDHomogeneous

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

T0 = 0.0
T1 = 1.0
SIGMA = 0.2
GEOMETRY = SPDHomogeneous(2)
X0 = jnp.array([[1.4, 0.2], [0.2, 0.9]], dtype=jnp.float64)
SO3_X0 = jnp.eye(3, dtype=jnp.float64)

_ees25_recurrence = LowStorageRecurrence(
    A=np.array([-0.5, -2.0]),
    B=np.array([0.5, 1.0, 0.25]),
    C=np.array([0.0, 0.5, 1.0]),
)

_ees27_recurrence = LowStorageRecurrence(
    A=np.array([1.0 - np.sqrt(2.0), -1.0, -(1.0 + np.sqrt(2.0))]),
    B=np.array(
        [
            0.5 * (2.0 - np.sqrt(2.0)),
            0.5 * np.sqrt(2.0),
            0.5 * np.sqrt(2.0),
            0.25 * (2.0 - np.sqrt(2.0)),
        ]
    ),
    C=np.array(
        [
            0.0,
            0.5 * (2.0 - np.sqrt(2.0)),
            0.5 * np.sqrt(2.0),
            1.0,
        ]
    ),
)


class EES25(LowStorageSolver):
    recurrance = _ees25_recurrence

    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 5


class EES27(LowStorageSolver):
    recurrance = _ees27_recurrence

    def order(self, terms):
        del terms
        return 2

    def antisymmetric_order(self, terms):
        del terms
        return 7


class SO3ExactExp(LieGroup):
    @property
    def lie_algebra_dimension(self) -> int:
        return 3

    def frame(self, x: Array) -> Array:
        basis = jnp.stack(
            [
                jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
                jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
                jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            ],
            axis=-1,
        )
        return jnp.einsum("ab,bck->ack", x, basis.astype(x.dtype))

    def to_frame(self, x: Array, v: Array) -> Array:
        omega = x.T @ v
        return jnp.array([omega[2, 1], omega[0, 2], omega[1, 0]], dtype=v.dtype)

    def from_frame(self, x: Array, a: Array) -> Array:
        a = jnp.asarray(a, dtype=x.dtype)
        omega = jnp.array(
            [
                [0.0, -a[2], a[1]],
                [a[2], 0.0, -a[0]],
                [-a[1], a[0], 0.0],
            ],
            dtype=x.dtype,
        )
        return x @ omega

    def frozen_flow(self, x: Array, a: Array) -> Array:
        a = jnp.asarray(a, dtype=x.dtype)
        omega = jnp.array(
            [
                [0.0, -a[2], a[1]],
                [a[2], 0.0, -a[0]],
                [-a[1], a[0], 0.0],
            ],
            dtype=x.dtype,
        )
        theta_sq = jnp.sum(omega * omega) / 2.0
        theta = jnp.sqrt(theta_sq)
        sin_over_theta = jnp.where(
            theta_sq > 1e-16,
            jnp.sin(theta) / theta,
            1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        )
        one_minus_cos_over_theta_sq = jnp.where(
            theta_sq > 1e-16,
            (1.0 - jnp.cos(theta)) / theta_sq,
            0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        )
        ident = jnp.eye(3, dtype=x.dtype)
        exp_omega = (
            ident
            + sin_over_theta * omega
            + one_minus_cos_over_theta_sq * (omega @ omega)
        )
        return x @ exp_omega

    def retraction(self, x: Array, v: Array) -> Array:
        return self.frozen_flow(x, self.to_frame(x, v))

    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        del a
        return b


SO3_GEOMETRY = SO3ExactExp()


def sym(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def spd_sqrt(matrix: Array) -> Array:
    matrix = sym(matrix)
    evals, evecs = jnp.linalg.eigh(matrix)
    evals = jnp.clip(evals, min=jnp.finfo(matrix.dtype).eps)
    return (evecs * jnp.sqrt(evals)) @ evecs.T


def spd_sym_lift_matrices(x: Array) -> tuple[Array, Array]:
    x = sym(jnp.asarray(x))
    s1 = jnp.array(
        [
            [0.55 + 0.18 * x[0, 0], 0.20 + 0.12 * x[0, 1]],
            [0.20 + 0.12 * x[0, 1], -0.25 + 0.10 * x[1, 1]],
        ],
        dtype=x.dtype,
    )
    s2 = jnp.array(
        [
            [0.15 + 0.22 * x[0, 1], -0.30 + 0.14 * x[0, 0]],
            [-0.30 + 0.14 * x[0, 0], 0.65 + 0.16 * x[1, 1]],
        ],
        dtype=x.dtype,
    )
    return SIGMA * sym(s1), SIGMA * sym(s2)


def spd_homogeneous_columns(x: Array) -> Array:
    s1, s2 = spd_sym_lift_matrices(x)
    col1 = GEOMETRY.from_frame(x, s1.reshape(-1))
    col2 = GEOMETRY.from_frame(x, s2.reshape(-1))
    return jnp.stack((col1, col2), axis=-1)


def spd_coeffs_prod(t: RealScalarLike, x: Array, args: Args, control: Array) -> Array:
    del t, args
    s1, s2 = spd_sym_lift_matrices(x)
    coeff_matrix = control[0] * s1 + control[1] * s2
    return coeff_matrix.reshape(-1)


def vector_field(t, x, args):
    del t, args
    return spd_homogeneous_columns(x)


def so3_vector_field(t: RealScalarLike, x: Array, args: Args) -> Array:
    del t, args
    coeffs = SIGMA * jnp.array(
        [
            [0.9 + 0.2 * x[0, 0], 0.15 + 0.25 * x[0, 1]],
            [0.25 + 0.2 * x[1, 2], -0.35 + 0.2 * x[1, 1]],
            [0.1 + 0.3 * x[2, 0], 0.8 + 0.15 * x[2, 2]],
        ],
        dtype=x.dtype,
    )
    return jnp.einsum("ija,ab->ijb", SO3_GEOMETRY.frame(x), coeffs)


def euclidean_vector_field(t, y, args):
    del t, args
    return jnp.stack([jnp.cos(y), jnp.sin(y)], axis=-1)


def get_2d_bm(num_steps: int, length: float, key: jax.Array) -> np.ndarray:
    dt = length / num_steps
    dw = np.asarray(
        jax.random.normal(key, (num_steps, 2), dtype=jnp.float64) * np.sqrt(dt)
    )
    x = np.zeros((num_steps + 1, 2), dtype=np.float64)
    x[1:] = np.cumsum(dw, axis=0)
    return x


def make_matrix_method_runner(
    solver,
    *,
    geometry,
    vector_field_fn: Callable[[RealScalarLike, Array, Args], Array],
    y0_default: np.ndarray,
    coeffs_prod_fn: Callable[[RealScalarLike, Array, Args, Array], Array] | None = None,
) -> Callable[..., np.ndarray]:
    stepsize_controller = diffrax.ConstantStepSize()
    adjoint = (
        ReversibleAdjoint()
        if isinstance(solver, AbstractReversibleSolver)
        else diffrax.RecursiveCheckpointAdjoint()
    )

    @lru_cache(maxsize=None)
    def _compiled_runner(num_points: int):
        if num_points < 2:
            raise ValueError("Need at least two control points.")

        ts = jnp.linspace(T0, T1, num_points, dtype=jnp.float64)
        dt0 = float((T1 - T0) / (num_points - 1))
        saveat = diffrax.SaveAt(ts=ts)

        @jax.jit
        def _run(x: jax.Array, y0: jax.Array) -> jax.Array:
            term = GeometricTerm(
                inner=diffrax.ControlTerm(
                    vector_field=vector_field_fn,
                    control=diffrax.LinearInterpolation(ts=ts, ys=x),
                ),
                geometry=geometry,
                coeffs_prod_fn=coeffs_prod_fn,
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


def make_method_runner(solver) -> Callable[..., np.ndarray]:
    return make_matrix_method_runner(
        solver,
        geometry=GEOMETRY,
        vector_field_fn=vector_field,
        y0_default=np.asarray(X0, dtype=np.float64),
        coeffs_prod_fn=spd_coeffs_prod,
    )


def make_so3_method_runner(solver) -> Callable[..., np.ndarray]:
    return make_matrix_method_runner(
        solver,
        geometry=SO3_GEOMETRY,
        vector_field_fn=so3_vector_field,
        y0_default=np.asarray(SO3_X0, dtype=np.float64),
    )


def make_euclidean_method_runner(solver) -> Callable[..., np.ndarray]:
    stepsize_controller = diffrax.ConstantStepSize()
    adjoint = (
        ReversibleAdjoint()
        if isinstance(solver, AbstractReversibleSolver)
        else diffrax.RecursiveCheckpointAdjoint()
    )

    @lru_cache(maxsize=None)
    def _compiled_runner(num_points: int):
        if num_points < 2:
            raise ValueError("Need at least two control points.")

        ts = jnp.linspace(T0, T1, num_points, dtype=jnp.float64)
        dt0 = float((T1 - T0) / (num_points - 1))
        saveat = diffrax.SaveAt(ts=ts)

        @jax.jit
        def _run(x: jax.Array, y0: jax.Array) -> jax.Array:
            term = diffrax.ControlTerm(
                vector_field=euclidean_vector_field,
                control=diffrax.LinearInterpolation(ts=ts, ys=x),
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
                adjoint=adjoint,
                max_steps=num_points + 4,
                throw=True,
            )
            assert solution.ys is not None
            return solution.ys

        return _run

    def _run(x: np.ndarray, y0: float | None = None) -> np.ndarray:
        x_arr = jnp.asarray(np.asarray(x, dtype=np.float64), dtype=jnp.float64)
        runner = _compiled_runner(int(x_arr.shape[0]))
        if y0 is None:
            y0 = 1.0
        y = runner(x_arr, jnp.asarray(y0, dtype=jnp.float64))
        return np.asarray(y, dtype=np.float64).reshape(-1)

    return _run


def get_error(y_exact: np.ndarray, y_vals: np.ndarray, step: int) -> float:
    true_vals = y_exact[::step]
    return float(np.max(np.linalg.norm(true_vals - y_vals, axis=(1, 2))))


def get_scalar_error(y_exact: np.ndarray, y_vals: np.ndarray, step: int) -> float:
    true_vals = y_exact[::step]
    return float(np.max(np.abs(true_vals - y_vals)))


def matrix_distance(y0: np.ndarray, y1: np.ndarray) -> float:
    return float(np.linalg.norm(y0 - y1))


def scalar_distance(y0: np.ndarray | float, y1: np.ndarray | float) -> float:
    return abs(float(y0) - float(y1))


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
    err_label = (
        r"$\log_{10}(\mathcal{E}(h))$"
        if not backward
        else r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"
    )
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
    ax.set_xlabel(r"$\log_{10}(h)$")
    ax.set_ylabel(err_label)


def expected_rates(
    solver,
    hurst: float,
) -> tuple[float, float]:
    order_fn = getattr(solver, "order", None)
    antisymmetric = getattr(solver, "antisymmetric_order", None)
    order = None
    antisymmetric_order = None
    if callable(order_fn):
        try:
            order = order_fn(None)
        except TypeError:
            order = order_fn()
    if callable(antisymmetric):
        try:
            antisymmetric_order = antisymmetric(None)
        except TypeError:
            antisymmetric_order = antisymmetric()
    if antisymmetric_order is None:
        assert order is not None
        antisymmetric_order = order + 1
    assert antisymmetric_order is not None
    return 2.0 * hurst - 0.5, float(antisymmetric_order + 1) * hurst - 1.0


def make_paths(num_paths: int, path_power: int, key: jax.Array) -> list[np.ndarray]:
    num_steps = 2**path_power
    keys = jax.random.split(key, num_paths)
    return [get_2d_bm(num_steps, T1 - T0, subkey) for subkey in keys]


def plot_grid(
    paths: list[np.ndarray],
    hs: np.ndarray,
    hurst: float,
    output_dir: Path,
) -> Path:
    benchmarks = [
        (
            "SPD(2) manifold CF-EES(2,5;1/4)",
            CFEES25(),
            make_method_runner,
            get_error,
            matrix_distance,
        ),
        (
            "SPD(2) manifold CF-EES(2,7;1/4)",
            CFEES27(),
            make_method_runner,
            get_error,
            matrix_distance,
        ),
        (
            "SO(3) manifold CF-EES(2,5;1/4)",
            CFEES25(),
            make_so3_method_runner,
            get_error,
            matrix_distance,
        ),
        (
            "SO(3) manifold CF-EES(2,7;1/4)",
            CFEES27(),
            make_so3_method_runner,
            get_error,
            matrix_distance,
        ),
        (
            "R manifold EES(2,5)",
            EES25(),
            make_euclidean_method_runner,
            get_scalar_error,
            scalar_distance,
        ),
        (
            "R manifold EES(2,7)",
            EES27(),
            make_euclidean_method_runner,
            get_scalar_error,
            scalar_distance,
        ),
    ]

    fig, axes = plt.subplots(len(benchmarks), 2, figsize=(10, 3.2 * len(benchmarks)))
    titles = ["Forward", "Backward"]

    for i, (name, solver, runner_factory, error_fn, distance_fn) in enumerate(
        benchmarks
    ):
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
        axes[i][0].set_title(f"{name} {titles[0]}")

        plot_curve(
            name,
            hs,
            backward_y,
            backward_rate,
            axes[i][1],
            backward=True,
        )
        axes[i][1].set_title(f"{name} {titles[1]}")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stochastic_convergence.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--path-power", type=int, default=13)
    parser.add_argument("--hurst", type=float, default=0.5)
    parser.add_argument("--min-power", type=int, default=2)
    parser.add_argument("--max-power", type=int, default=9)
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
    paths = make_paths(args.num_paths, args.path_power, jax.random.key(args.seed))
    output_path = plot_grid(paths, hs, args.hurst, args.output_dir)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
