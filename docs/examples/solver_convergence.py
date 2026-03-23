"""Convergence experiment for the georax solvers."""

from __future__ import annotations

import argparse
import os
import sys
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
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array

from georax import CFEES25, CG2, CG4, GeometricTerm, LieGroup

matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

T0 = 0.0
T1 = 1.0
MAX_STEPS = 100_000


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

    def retraction(self, x: Array, v: Array) -> Array:
        omega = x.T @ v
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

    def chart_differential_inv(self, a: Array, b: Array) -> Array:
        del a
        return b


def omega_body(t: RealScalarLike) -> Array:
    t_array = jnp.asarray(t)
    return jnp.array(
        [
            0.8 + 0.45 * jnp.sin(0.7 * t_array),
            0.55 * jnp.cos(1.3 * t_array + 0.2),
            0.35 + 0.6 * jnp.sin(0.9 * t_array - 0.4),
        ]
    )


def vector_field(t: RealScalarLike, y: Array, args: Args) -> Array:
    del args
    omega = omega_body(t)
    skew = jnp.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    return y @ skew


def make_term() -> GeometricTerm:
    return GeometricTerm(inner=diffrax.ODETerm(vector_field), geometry=SO3ExactExp())


def reference_solution(term: GeometricTerm, y0: Array) -> Array:
    solution = diffrax.diffeqsolve(
        term.inner,
        diffrax.Dopri8(),
        T0,
        T1,
        1e-3,
        y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12),
        max_steps=MAX_STEPS,
        throw=True,
    )
    assert solution.ys is not None
    return solution.ys[0]


def forward_error(
    term: GeometricTerm,
    solver,
    dt: float,
    y0: Array,
    y_exact: Array,
) -> float:
    solution = diffrax.diffeqsolve(
        term,
        solver,
        T0,
        T1,
        dt,
        y0,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=MAX_STEPS,
        throw=True,
    )
    assert solution.ys is not None
    return float(jnp.linalg.norm(solution.ys[0] - y_exact))


def reversible_roundtrip_error(
    term: GeometricTerm,
    solver: diffrax.AbstractReversibleSolver,
    dt: float,
    y0: Array,
) -> float:
    num_steps = round((T1 - T0) / dt)
    ts = [T0 + i * dt for i in range(num_steps + 1)]
    y = y0
    solver_state = solver.init(term, ts[0], ts[1], y, None)

    for i in range(num_steps):
        y, _, _, solver_state, result = solver.step(
            term, ts[i], ts[i + 1], y, None, solver_state, False
        )
        assert result == diffrax.RESULTS.successful

    for i in range(num_steps, 0, -1):
        tm1 = ts[i - 2] if i - 2 >= 0 else ts[i - 1]
        y, _, solver_state, result = solver.backward_step(
            term, ts[i - 1], ts[i], y, None, (tm1,), solver_state, False
        )
        assert result == diffrax.RESULTS.successful

    return float(jnp.linalg.norm(y - y0))


def backward_error(
    term: GeometricTerm,
    solver,
    dt: float,
    y0: Array,
    y_exact: Array,
) -> float:
    if isinstance(solver, diffrax.AbstractReversibleSolver):
        return reversible_roundtrip_error(term, solver, dt, y0)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        T1,
        T0,
        -dt,
        y_exact,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=MAX_STEPS,
        throw=True,
    )
    assert solution.ys is not None
    return float(jnp.linalg.norm(solution.ys[0] - y0))


def plot(
    solver,
    name: str,
    term: GeometricTerm,
    y0: Array,
    y_exact: Array,
    hs: list[float],
    ax,
    *,
    backward: bool = False,
) -> None:
    error_fn = backward_error if backward else forward_error
    errors = [error_fn(term, solver, h, y0, y_exact) for h in hs]
    x = np.log10(np.asarray(hs, dtype=np.float64))
    y = np.log10(np.maximum(np.asarray(errors, dtype=np.float64), np.finfo(np.float64).tiny))
    dx = np.array([x[0], x[-1]], dtype=np.float64)

    if backward and isinstance(solver, diffrax.AbstractReversibleSolver):
        slope = float(solver.antisymmetric_order(term))
    else:
        slope = float(solver.order(term))

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


def plot_grid(hs: list[float], output_dir: Path) -> Path:
    term = make_term()
    y0 = jnp.eye(3, dtype=jnp.float64)
    y_exact = reference_solution(term, y0)
    solvers = [
        ("CG2", CG2()),
        ("CG4", CG4()),
        ("CF-EES(2,5;1/4)", CFEES25()),
    ]

    fig, axes = plt.subplots(len(solvers), 2, figsize=(10, 3.2 * len(solvers)))
    titles = ["Forward", "Backward"]

    for i, (name, solver) in enumerate(solvers):
        for j in range(2):
            plot(
                solver,
                name,
                term,
                y0,
                y_exact,
                hs,
                axes[i][j],
                backward=bool(j),
            )
            axes[i][j].set_title(f"{name} {titles[j]}")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "solver_convergence.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-power",
        type=int,
        default=2,
        help="Smallest power in h = 2^(-k).",
    )
    parser.add_argument(
        "--max-power",
        type=int,
        default=8,
        help="Largest power in h = 2^(-k).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hs = [2.0 ** (-k) for k in range(args.min_power, args.max_power + 1)]
    output_path = plot_grid(hs, args.output_dir)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
