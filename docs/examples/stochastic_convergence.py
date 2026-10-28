"""Decimal pathwise SO(3) convergence experiments for the CF-EES solvers.

Run: python docs/examples/stochastic_convergence.py --precision 80

Gaussian/FFT path samples are generated in binary64 and converted exactly to
Decimal before integration. Stages, charts, errors and log averaging use Decimal.
Forward errors compare nested coarse paths with a finite fine-grid reference;
round trips traverse each coarse path in reverse from its computed endpoint.
The full default ensemble is substantially slower than the JAX experiment.
Each Hurst value has a single-column figure, with forward then round-trip
errors for each method in order.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
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
    identity,
    make_methods,
    norm,
    stochastic_coeffs,
)
from fbm import FBM


def make_paths(
    num_paths: int, path_power: int, seed: int, hurst: Decimal
) -> list[Array]:
    """Generate once on the finest grid; all coarse paths are nested slices."""
    count = 2**path_power
    rng = np.random.default_rng(seed)
    if hurst == Decimal("0.5"):
        scale = (Decimal(1) / count).sqrt()
        paths = []
        for _ in range(num_paths):
            normals = rng.standard_normal((count, 2))
            increments = np.array(
                [[Decimal.from_float(float(v)) * scale for v in row] for row in normals]
            )
            path = np.empty((count + 1, 2), dtype=object)
            path[0] = Decimal(0)
            path[1:] = np.cumsum(increments, axis=0)
            paths.append(path)
        return paths

    # fbm uses NumPy's legacy global RNG; restore it after seeded sampling.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        sampler = FBM(n=count, hurst=float(hurst), length=1, method="daviesharte")
        return [
            np.array(
                [
                    [Decimal.from_float(float(v)) for v in row]
                    for row in np.stack([sampler.fbm(), sampler.fbm()], axis=-1)
                ]
            )
            for _ in range(num_paths)
        ]
    finally:
        np.random.set_state(state)


def solve_path(
    method: DecimalMethod, path: Array, y0: Array | None = None
) -> list[Array]:
    y = identity() if y0 is None else y0
    values = [y]
    h = Decimal(1) / (len(path) - 1)
    for i, dw in enumerate(np.diff(path, axis=0)):
        y = method.step(
            i * h, h, y, lambda _, state, dw=dw: stochastic_coeffs(state) @ dw
        )
        values.append(y)
    return values


def compute_error_curves(
    method: DecimalMethod, paths: list[Array], powers: list[int]
) -> tuple[list[list[Decimal]], list[list[Decimal]]]:
    forward, backward = [], []
    for path_index, path in enumerate(paths):
        print(f"{method.name}: path {path_index + 1}/{len(paths)}", flush=True)
        reference = solve_path(method, path)
        path_forward, path_backward = [], []
        for power in powers:
            stride = (len(path) - 1) // 2**power
            coarse_path = path[::stride]
            coarse = solve_path(method, coarse_path)
            path_forward.append(
                max(
                    norm(exact - y)
                    for exact, y in zip(reference[::stride], coarse, strict=True)
                )
            )
            reverse = solve_path(method, coarse_path[::-1], coarse[-1])
            path_backward.append(norm(reverse[-1] - identity()))
        forward.append(path_forward)
        backward.append(path_backward)
    return forward, backward


def geometric_mean_errors(
    samples: list[list[Decimal]], precision: int
) -> list[Decimal]:
    floor = Decimal(10) ** (1 - precision)
    return [
        Decimal(10)
        ** (
            sum((max(error, floor).log10() for error in errors), Decimal(0))
            / len(errors)
        )
        for errors in zip(*samples, strict=True)
    ]


def expected_rates(method: DecimalMethod, hurst: Decimal) -> tuple[Decimal, Decimal]:
    assert method.antisymmetric_order is not None
    return 2 * hurst - Decimal("0.5"), (method.antisymmetric_order + 1) * hurst - 1


def plot_grid(
    paths: list[Array],
    min_power: int,
    max_power: int,
    hurst: Decimal,
    precision: int,
    seed: int,
    output_dir: Path,
) -> Path:
    with localcontext() as ctx:
        ctx.prec = precision
        methods = [m for m in make_methods() if m.antisymmetric_order is not None]
        powers = list(range(min_power, max_power + 1))
        hs = [Decimal(1) / 2**power for power in powers]
        fig, axes = plt.subplots(2 * len(methods), 1, figsize=(5, 6.4 * len(methods)))
        solver_results: dict[str, object] = {}
        results: dict[str, object] = {
            "arithmetic": "decimal.Decimal",
            "precision_digits": precision,
            "equation": "dR = R skew(F(R) dX), two-channel SO(3) diffusion, sigma=1",
            "t0": "0",
            "t1": "1",
            "y0": "I_3",
            "hurst": str(hurst),
            "seed": seed,
            "num_paths": len(paths),
            "path_steps": len(paths[0]) - 1,
            "path_sampling": "binary64 Gaussian samples / Davies-Harte FFT, converted with Decimal.from_float; Brownian scaling and accumulation use Decimal",
            "reference": "same method on the common finest sampled path (finite-grid reference)",
            "forward_error": "geometric mean of pathwise maximum Frobenius errors at coarse nodes",
            "roundtrip_error": "geometric mean of endpoint Frobenius errors after reversing the coarse path",
            "resolved_rule": "every sample error exceeds the Decimal roundoff threshold",
            "min_power": min_power,
            "max_power": max_power,
            "h": [str(h) for h in hs],
            "solvers": solver_results,
        }
        for i, method in enumerate(methods):
            forward, backward = compute_error_curves(method, paths, powers)
            rates = expected_rates(method, hurst)
            method_results = method.metadata()
            for j, (mode, samples, rate) in enumerate(
                zip(("forward", "roundtrip"), (forward, backward), rates, strict=True)
            ):
                method_results[mode] = plot_curve(
                    method.name,
                    hs,
                    geometric_mean_errors(samples, precision),
                    float(rate),
                    precision,
                    axes[2 * i + j],
                    backward=bool(j),
                    sample_errors=samples,
                )
                set_method_title(axes[2 * i + j], method.title, backward=bool(j))
            solver_results[method.name] = method_results
        # Decimal formatting keeps distinct H values distinct in output names.
        label = format((hurst * 100).normalize(), "f")
        return save_figure(
            fig, results, output_dir, f"stochastic_convergence_decimal_H{label}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, min_power=3, max_power=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-paths", type=int, default=12)
    parser.add_argument("--path-power", type=int, default=13)
    parser.add_argument(
        "--hurst",
        type=Decimal,
        nargs="+",
        default=list(map(Decimal, ("0.4", "0.5", "0.6"))),
    )
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    if args.num_paths < 1:
        parser.error("num-paths must be positive.")
    if args.path_power <= args.max_power:
        parser.error(
            "path-power must exceed max-power so the reference is finer than every plotted path."
        )
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must lie in [0, 2^32).")
    if any(not h.is_finite() or not Decimal("0.25") < h < 1 for h in args.hurst):
        parser.error("Each Hurst parameter must satisfy 1/4 < H < 1.")
    return args


def main() -> None:
    args = parse_args()
    with localcontext() as ctx:
        ctx.prec = args.precision
        for hurst in args.hurst:
            paths = make_paths(args.num_paths, args.path_power, args.seed, hurst)
            output_path = plot_grid(
                paths,
                args.min_power,
                args.max_power,
                hurst,
                args.precision,
                args.seed,
                args.output_dir,
            )
            print(f"H={hurst:g} saved {output_path} (also .pdf and .json)")


if __name__ == "__main__":
    main()
