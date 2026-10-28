"""Decimal SO(3) convergence experiments, with PNG, PDF, and JSON output.

Run: python docs/examples/solver_convergence.py --precision 80

The stage formulas and charts are evaluated independently of JAX in Decimal.
Forward errors use a Decimal power-series reference for the original SO(3)
accuracy problem. CF-EES round trips start from the computed forward endpoint;
CG2/CG4 backward solves start from the reference, as in the original example.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from pathlib import Path

from _convergence_plot import (
    add_common_arguments,
    plot_curve,
    plt,
    save_figure,
    set_method_title,
    validate_common_arguments,
)
from _decimal_so3 import (
    Array,
    DecimalMethod,
    deterministic_coeffs,
    identity,
    make_methods,
    norm,
    reference_solution,
)


def solve_roundtrip(
    method: DecimalMethod, num_steps: int, backward_y0: Array | None = None
) -> tuple[Array, Array]:
    h = Decimal(1) / num_steps
    y = identity()
    for i in range(num_steps):
        y = method.step(i * h, h, y, lambda t, _: h * deterministic_coeffs(t))
    forward_y = y
    if backward_y0 is not None:
        y = backward_y0
    for i in range(num_steps, 0, -1):
        y = method.step(i * h, -h, y, lambda t, _: -h * deterministic_coeffs(t))
    return forward_y, y


def plot_grid(
    min_power: int,
    max_power: int,
    precision: int,
    output_dir: Path,
    *,
    include_cg: bool = False,
) -> Path:
    with localcontext() as ctx:
        ctx.prec = precision
        methods = [
            m for m in make_methods() if include_cg or m.antisymmetric_order is not None
        ]
        counts = [2**power for power in range(min_power, max_power + 1)]
        hs = [Decimal(1) / count for count in counts]
        exact = reference_solution()
        fig, axes = plt.subplots(len(methods), 2, figsize=(10, 3.2 * len(methods)))
        solver_results: dict[str, object] = {}
        results: dict[str, object] = {
            "arithmetic": "decimal.Decimal",
            "precision_digits": precision,
            "equation": "R' = R skew(-omega_z, omega_y, -omega_x)",
            "omega": [
                "0.8 + 0.45 sin(0.7 t)",
                "0.55 cos(1.3 t + 0.2)",
                "0.35 + 0.6 sin(0.9 t - 0.4)",
            ],
            "t0": "0",
            "t1": "1",
            "y0": "I_3",
            "error_norm": "Frobenius",
            "reference": "Decimal power series with ten guard digits",
            "reference_y1": [[str(x) for x in row] for row in exact],
            "min_power": min_power,
            "max_power": max_power,
            "h": [str(h) for h in hs],
            "solvers": solver_results,
        }
        for i, method in enumerate(methods):
            print(f"Computing {method.name} at {precision} digits...", flush=True)
            backward_y0 = exact if method.antisymmetric_order is None else None
            values = [solve_roundtrip(method, count, backward_y0) for count in counts]
            forward_errors = [norm(y - exact) for y, _ in values]
            roundtrip_errors = [norm(y - identity()) for _, y in values]
            method_results = method.metadata()
            reverse_mode = "roundtrip" if method.antisymmetric_order else "backward"
            method_results["backward_initial_value"] = (
                "computed forward endpoint"
                if method.antisymmetric_order
                else "reference endpoint"
            )
            if method.name == "CG4":
                method_results["coefficient_source"] = (
                    "Decimal refinement of 13 CG order conditions; a21=0.8177227988124852 and a31=0.3199876375476427 fixed exactly"
                )
            for j, (mode, errors, rate) in enumerate(
                (
                    ("forward", forward_errors, method.order),
                    (
                        reverse_mode,
                        roundtrip_errors,
                        method.antisymmetric_order or method.order,
                    ),
                )
            ):
                method_results[mode] = plot_curve(
                    method.name,
                    hs,
                    errors,
                    rate,
                    precision,
                    axes[i, j],
                    backward=bool(j),
                    mode=mode,
                )
                set_method_title(axes[i, j], method.title, backward=bool(j))
            solver_results[method.name] = method_results
        stem = (
            "solver_convergence_decimal_with_cg"
            if include_cg
            else "solver_convergence_decimal"
        )
        return save_figure(fig, results, output_dir, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, min_power=2, max_power=12)
    parser.add_argument(
        "--include-cg",
        action="store_true",
        help="Also plot CG2 and CG4, including backward solves from the reference.",
    )
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    return args


def main() -> None:
    args = parse_args()
    output_path = plot_grid(
        args.min_power,
        args.max_power,
        args.precision,
        args.output_dir,
        include_cg=args.include_cg,
    )
    print(f"Saved {output_path} (also .pdf and .json)")


if __name__ == "__main__":
    main()
