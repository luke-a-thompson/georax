"""Shared Decimal convergence charts, following diffrax-lowstorage's examples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.markers as mkr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

XLABEL = r"$\log_{10}(h)$"
YLABEL_FORWARD = r"$\log_{10}(\mathcal{E}(h))$"
YLABEL_BACKWARD = r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$"


def set_method_title(ax: Axes, method: str, *, backward: bool = False) -> None:
    error = r"\overleftarrow{\mathcal{E}}(h)" if backward else r"\mathcal{E}(h)"
    ax.set_title(rf"${error}$ for ${method}$")


def roundoff_thresholds(hs: Sequence[Decimal], precision: int) -> list[Decimal]:
    unit_roundoff = Decimal(10) ** (1 - precision)
    return [100 * unit_roundoff / h for h in hs]


def plot_curve(
    name: str,
    hs: Sequence[Decimal],
    errors: Sequence[Decimal],
    expected_slope: float,
    precision: int,
    ax: Axes,
    *,
    backward: bool = False,
    mode: str | None = None,
    sample_errors: Sequence[Sequence[Decimal]] | None = None,
) -> dict[str, object]:
    """Fit resolved log-errors only and retain Decimal values in JSON output."""
    unit_roundoff = Decimal(10) ** (1 - precision)
    thresholds = roundoff_thresholds(hs, precision)
    samples = [errors] if sample_errors is None else sample_errors
    resolved = np.array(
        [
            all(sample[i] > cutoff for sample in samples)
            for i, cutoff in enumerate(thresholds)
        ]
    )
    x = np.array([float(h.log10()) for h in hs])
    y = np.array([float(max(error, unit_roundoff).log10()) for error in errors])
    measured = (
        float(np.polyfit(x[resolved], y[resolved], 1)[0])
        if resolved.sum() >= 2
        else None
    )
    label = YLABEL_BACKWARD if backward else YLABEL_FORWARD
    ax.scatter(
        x[resolved],
        y[resolved],
        marker=mkr.MarkerStyle("x", fillstyle="none"),
        color="crimson",
        label=label,
    )
    if resolved.any():
        dx = np.array([x[resolved].min(), x[resolved].max()])
        intercept = float(np.mean(y[resolved]) - expected_slope * np.mean(x[resolved]))
        ax.plot(
            dx,
            expected_slope * dx + intercept,
            color="mediumblue",
            label=f"{expected_slope:.1f}$x + c$",
        )
    if (~resolved).any():
        ax.scatter(
            x[~resolved],
            y[~resolved],
            marker="x",
            color="0.65",
            label="Roundoff",
        )
    ax.legend()
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(label)
    mode = mode or ("round-trip" if backward else "forward")
    fit_text = "unresolved" if measured is None else f"{measured:.6f}"
    print(
        f"{name} {mode} slope: {fit_text} (expected {expected_slope:g}; {resolved.sum()}/{len(hs)} resolved points)",
        flush=True,
    )
    result: dict[str, object] = {
        "expected_slope": expected_slope,
        "measured_slope": measured,
        "fit_points": int(resolved.sum()),
        "resolved": resolved.tolist(),
        "errors": [str(error) for error in errors],
        "log10_errors": y.tolist(),
        "roundoff_thresholds": [str(cutoff) for cutoff in thresholds],
    }
    if sample_errors is not None:
        result["sample_errors"] = [
            [str(error) for error in sample] for sample in samples
        ]
    return result


def save_figure(
    fig: Figure, results: dict[str, object], output_dir: Path, stem: str
) -> Path:
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    path.with_suffix(".json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    return path


def add_common_arguments(
    parser: argparse.ArgumentParser, *, min_power: int, max_power: int
) -> None:
    parser.add_argument(
        "--precision",
        type=int,
        default=80,
        help="Decimal precision in digits (default: 80).",
    )
    parser.add_argument(
        "--min-power", type=int, default=min_power, help="Smallest k in h = 2^(-k)."
    )
    parser.add_argument(
        "--max-power", type=int, default=max_power, help="Largest k in h = 2^(-k)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )


def validate_common_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.precision < 16:
        parser.error("precision must be at least 16 decimal digits.")
    if args.min_power < 0 or args.max_power < args.min_power:
        parser.error("Require 0 <= min-power <= max-power.")
