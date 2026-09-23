"""Free-nilpotent antisymmetry experiment for CF-EES(2,5).

This script tests whether the observed antisymmetric order of ``CFEES25`` is
still visible on a truncated free nilpotent Lie group, where no geometry-
specific low-order identities are available beyond the universal Lie relations
and the truncation itself.
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.markers as mkr
import matplotlib.pyplot as plt
import numpy as np

MAX_DEGREE = 7
DEFAULT_POWERS = tuple(range(0, 6))
FORWARD_T1 = 1.0
REFERENCE_REFINEMENT = 32

_A = np.array([-7.0 / 15.0, -35.0 / 32.0], dtype=np.float64)
_B = np.array([1.0 / 3.0, 15.0 / 16.0, 2.0 / 5.0], dtype=np.float64)


Word = tuple[int, ...]
Series = list[np.ndarray]


def _all_words(rank: int, degree: int) -> list[Word]:
    return [tuple(word) for word in itertools.product(range(rank), repeat=degree)]


def _is_lyndon(word: Word) -> bool:
    if len(word) == 0:
        return False
    return all(word < word[i:] for i in range(1, len(word)))


def _standard_factorization(word: Word) -> tuple[Word, Word]:
    for split in range(1, len(word)):
        suffix = word[split:]
        if _is_lyndon(suffix):
            return word[:split], suffix
    raise ValueError(f"No Lyndon factorization found for {word}.")


def _zeros_series(rank: int, step: int) -> Series:
    series: Series = [np.zeros(1, dtype=np.float64)]
    for degree in range(1, step + 1):
        series.append(np.zeros(rank**degree, dtype=np.float64))
    return series


def _identity_series(rank: int, step: int) -> Series:
    series = _zeros_series(rank, step)
    series[0][0] = 1.0
    return series


def _series_add_scaled(dst: Series, src: Series, scale: float) -> None:
    for degree in range(len(dst)):
        dst[degree] = dst[degree] + scale * src[degree]


def _series_scale(src: Series, scale: float) -> Series:
    return [scale * block for block in src]


def _series_mul(a: Series, b: Series, rank: int, step: int) -> Series:
    out = _zeros_series(rank, step)
    out[0][0] = a[0][0] * b[0][0]
    for degree in range(1, step + 1):
        accum = a[0][0] * b[degree] + b[0][0] * a[degree]
        for left_degree in range(1, degree):
            right_degree = degree - left_degree
            accum = accum + np.multiply.outer(a[left_degree], b[right_degree]).reshape(
                -1
            )
        out[degree] = accum
    return out


def _series_exp(x: Series, rank: int, step: int) -> Series:
    out = _identity_series(rank, step)
    term = _identity_series(rank, step)
    for n in range(1, step + 1):
        term = _series_mul(term, x, rank, step)
        _series_add_scaled(out, term, 1.0 / math.factorial(n))
    return out


def _series_log(x: Series, rank: int, step: int) -> Series:
    if abs(x[0][0] - 1.0) > 1e-12:
        raise ValueError("The logarithm expects a unit tensor series.")
    u = [block.copy() for block in x]
    u[0][0] -= 1.0
    out = _zeros_series(rank, step)
    term = [block.copy() for block in u]
    for n in range(1, step + 1):
        _series_add_scaled(out, term, ((-1.0) ** (n + 1)) / n)
        if n < step:
            term = _series_mul(term, u, rank, step)
    return out


@dataclass(frozen=True)
class FreeNilpotentLyndon:
    rank: int
    step: int
    lyndon_words: tuple[Word, ...]
    degree_slices: tuple[slice, ...]
    basis_to_tensor: tuple[np.ndarray, ...]
    tensor_to_basis: tuple[np.ndarray, ...]
    total_dimension: int

    @classmethod
    def build(cls, rank: int, step: int) -> "FreeNilpotentLyndon":
        if rank < 2:
            raise ValueError("The free nilpotent test requires rank >= 2.")
        if step < 1:
            raise ValueError("The nilpotent step must be positive.")

        words_by_degree = {
            degree: _all_words(rank, degree) for degree in range(1, step + 1)
        }
        lyndon_by_degree = {
            degree: [word for word in words_by_degree[degree] if _is_lyndon(word)]
            for degree in range(1, step + 1)
        }

        cache: dict[Word, np.ndarray] = {}

        def expand(word: Word) -> np.ndarray:
            if word in cache:
                return cache[word]
            degree = len(word)
            dim = rank**degree
            arr = np.zeros(dim, dtype=np.float64)
            if degree == 1:
                arr[word[0]] = 1.0
            else:
                left, right = _standard_factorization(word)
                arr = np.multiply.outer(expand(left), expand(right)).reshape(-1)
                arr -= np.multiply.outer(expand(right), expand(left)).reshape(-1)
            cache[word] = arr
            return arr

        lyndon_words: list[Word] = []
        degree_slices: list[slice] = [slice(0, 0)]
        basis_to_tensor: list[np.ndarray] = [np.zeros((1, 0), dtype=np.float64)]
        tensor_to_basis: list[np.ndarray] = [np.zeros((0, 1), dtype=np.float64)]
        offset = 0
        for degree in range(1, step + 1):
            words = lyndon_by_degree[degree]
            lyndon_words.extend(words)
            next_offset = offset + len(words)
            degree_slices.append(slice(offset, next_offset))
            if words:
                basis = np.column_stack([expand(word) for word in words])
                basis_to_tensor.append(basis)
                tensor_to_basis.append(np.linalg.pinv(basis))
            else:
                basis_to_tensor.append(np.zeros((rank**degree, 0), dtype=np.float64))
                tensor_to_basis.append(np.zeros((0, rank**degree), dtype=np.float64))
            offset = next_offset

        return cls(
            rank=rank,
            step=step,
            lyndon_words=tuple(lyndon_words),
            degree_slices=tuple(degree_slices),
            basis_to_tensor=tuple(basis_to_tensor),
            tensor_to_basis=tuple(tensor_to_basis),
            total_dimension=offset,
        )

    def coords_to_series(self, coords: np.ndarray) -> Series:
        if coords.shape != (self.total_dimension,):
            raise ValueError(
                f"Expected coordinate vector of shape ({self.total_dimension},), got {coords.shape}."
            )
        out = _zeros_series(self.rank, self.step)
        for degree in range(1, self.step + 1):
            block = coords[self.degree_slices[degree]]
            if block.size:
                out[degree] = self.basis_to_tensor[degree] @ block
        return out

    def series_to_coords(self, series: Series) -> np.ndarray:
        coords = np.zeros(self.total_dimension, dtype=np.float64)
        for degree in range(1, self.step + 1):
            if self.degree_slices[degree].stop > self.degree_slices[degree].start:
                coords[self.degree_slices[degree]] = (
                    self.tensor_to_basis[degree] @ series[degree]
                )
        return coords

    def bracket(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sx = self.coords_to_series(x)
        sy = self.coords_to_series(y)
        comm = _series_mul(sx, sy, self.rank, self.step)
        anti = _series_mul(sy, sx, self.rank, self.step)
        for degree in range(self.step + 1):
            comm[degree] -= anti[degree]
        return self.series_to_coords(comm)

    def bch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sx = self.coords_to_series(x)
        sy = self.coords_to_series(y)
        product = _series_mul(
            _series_exp(sx, self.rank, self.step),
            _series_exp(sy, self.rank, self.step),
            self.rank,
            self.step,
        )
        return self.series_to_coords(_series_log(product, self.rank, self.step))


@dataclass
class RandomVectorField:
    c: np.ndarray
    linear: np.ndarray
    q_left: np.ndarray
    q_right: np.ndarray
    q_out: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.c + self.linear @ x
        left = self.q_left @ x
        right = self.q_right @ x
        out = out + self.q_out @ (left * right)
        return out


@dataclass(frozen=True)
class AggregatedResult:
    rank: int
    hs_forward: np.ndarray
    forward_y: np.ndarray
    hs_backward: np.ndarray
    backward_y: np.ndarray
    backward_fit_count: int


def make_random_vector_field(
    algebra: FreeNilpotentLyndon, seed: int
) -> tuple[np.ndarray, RandomVectorField]:
    rng = np.random.default_rng(seed)
    dim = algebra.total_dimension
    basis_degrees = np.array(
        [len(word) for word in algebra.lyndon_words], dtype=np.float64
    )
    weights = 0.25 / basis_degrees

    x0 = rng.normal(size=dim) * weights
    c = rng.normal(size=dim) * weights
    linear = rng.normal(size=(dim, dim)) * (0.15 / max(dim, 1))

    q_rank = min(12, max(4, dim // 8))
    q_left = rng.normal(size=(q_rank, dim)) * (0.3 / math.sqrt(max(dim, 1)))
    q_right = rng.normal(size=(q_rank, dim)) * (0.3 / math.sqrt(max(dim, 1)))
    q_out = rng.normal(size=(dim, q_rank)) * (0.15 / max(q_rank, 1))

    field = RandomVectorField(
        c=c,
        linear=linear,
        q_left=q_left,
        q_right=q_right,
        q_out=q_out,
    )
    return x0, field


def cfees25_step(
    algebra: FreeNilpotentLyndon,
    field: RandomVectorField,
    x0: np.ndarray,
    h: float,
) -> np.ndarray:
    tmp = h * field(x0)
    y = algebra.bch(_B[0] * tmp, x0)

    coeffs = h * field(y)
    tmp = _A[0] * tmp + coeffs
    y = algebra.bch(_B[1] * tmp, y)

    coeffs = h * field(y)
    tmp = _A[1] * tmp + coeffs
    return algebra.bch(_B[2] * tmp, y)


def forward_backward_defect(
    algebra: FreeNilpotentLyndon,
    field: RandomVectorField,
    x0: np.ndarray,
    h: float,
) -> float:
    x1 = cfees25_step(algebra, field, x0, h)
    x2 = cfees25_step(algebra, field, x1, -h)
    defect = algebra.bch(x2, -x0)
    return float(np.linalg.norm(defect))


def integrate_cfees25(
    algebra: FreeNilpotentLyndon,
    field: RandomVectorField,
    x0: np.ndarray,
    h: float,
    t1: float,
) -> np.ndarray:
    steps = round(t1 / h)
    if abs(steps * h - t1) > 1e-12:
        raise ValueError(
            "The time horizon must be an integer multiple of the step size."
        )
    x = x0.copy()
    for _ in range(steps):
        x = cfees25_step(algebra, field, x, h)
    return x


def forward_order_error(
    algebra: FreeNilpotentLyndon,
    field: RandomVectorField,
    x0: np.ndarray,
    h: float,
    refinement: int,
) -> float:
    coarse = integrate_cfees25(algebra, field, x0, h, FORWARD_T1)
    fine = integrate_cfees25(algebra, field, x0, h / refinement, FORWARD_T1)
    return float(np.linalg.norm(coarse - fine))


def fitted_slope(step_sizes: np.ndarray, errors: np.ndarray) -> float:
    safe_errors = np.maximum(errors, np.finfo(np.float64).tiny)
    return float(np.polyfit(np.log(step_sizes), np.log(safe_errors), 1)[0])


def fitted_log_slope(step_sizes: np.ndarray, log_errors: np.ndarray) -> float:
    return float(np.polyfit(np.log10(step_sizes), log_errors, 1)[0])


def measurable_prefix_length(
    errors: np.ndarray, *, absolute_floor: float, min_ratio: float = 8.0
) -> int:
    if errors.size <= 2:
        return int(errors.size)
    usable = 1
    for index in range(1, errors.size):
        if errors[index] <= absolute_floor:
            break
        ratio = errors[index - 1] / errors[index]
        if ratio <= min_ratio:
            break
        usable += 1
    return max(3, min(usable, int(errors.size)))


def compute_error_curves(
    algebra: FreeNilpotentLyndon,
    seeds: list[int],
    hs: np.ndarray,
    refinement: int,
) -> tuple[np.ndarray, np.ndarray]:
    forward_y = np.zeros(len(hs) - 2, dtype=np.float64)
    backward_y = np.zeros(len(hs), dtype=np.float64)

    for seed in seeds:
        x0, field = make_random_vector_field(algebra, seed)
        backward_errors = np.array(
            [forward_backward_defect(algebra, field, x0, h) for h in hs],
            dtype=np.float64,
        )
        forward_errors = np.array(
            [forward_order_error(algebra, field, x0, h, refinement) for h in hs[:-2]],
            dtype=np.float64,
        )
        backward_y += np.log10(np.maximum(backward_errors, np.finfo(np.float64).tiny))
        forward_y += np.log10(np.maximum(forward_errors, np.finfo(np.float64).tiny))

    return forward_y / len(seeds), backward_y / len(seeds)


def plot_curve(
    name: str,
    h: np.ndarray,
    y: np.ndarray,
    slope: float,
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str,
    backward: bool = False,
    fit_count: int | None = None,
) -> None:
    x = np.log10(h)
    intercept = float(np.mean(y) - slope * np.mean(x))
    dx = np.array([x[0], x[-1]], dtype=np.float64)
    fit_x = x if fit_count is None else x[:fit_count]
    fit_y = y if fit_count is None else y[:fit_count]
    fit = np.polyfit(fit_x, fit_y, 1)
    mode = "backward" if backward else "forward"
    print(f"{name} {mode} slope: {fit[0]:.6f}")

    ax.scatter(
        x,
        y,
        marker=mkr.MarkerStyle("x", fillstyle="none"),
        color="crimson",
    )
    ax.plot(dx, slope * dx + intercept, color="mediumblue")
    ax.legend(["data", f"{slope:.1f}$x + c$"])
    ax.set_title(title)
    ax.set_xlabel(r"$\log_{10}(h)$")
    ax.set_ylabel(ylabel)


def save_plot_grid(
    results: list[AggregatedResult],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(len(results), 2, figsize=(10, 3.2 * len(results)))
    if len(results) == 1:
        axes = np.asarray([axes], dtype=object)

    for row, result in enumerate(results):
        label = f"rank={result.rank}"
        plot_curve(
            label,
            result.hs_forward,
            result.forward_y,
            2.0,
            axes[row][0],
            title=(
                rf"$\mathcal{{E}}(h)$ for $\mathrm{{CF\text{{-}}EES}}(2,5)$ "
                rf"on rank {result.rank}"
            ),
            ylabel=r"$\log_{10}(\mathcal{E}(h))$",
            backward=False,
        )
        plot_curve(
            label,
            result.hs_backward,
            result.backward_y,
            6.0,
            axes[row][1],
            title=(
                rf"$\overleftarrow{{\mathcal{{E}}}}(h)$ for "
                rf"$\mathrm{{CF\text{{-}}EES}}(2,5)$ on rank {result.rank}"
            ),
            ylabel=r"$\log_{10}(\overleftarrow{\mathcal{E}}(h))$",
            backward=True,
            fit_count=result.backward_fit_count,
        )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "free_nilpotent_antisymmetry.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Free-Lie ranks to test.",
    )
    parser.add_argument(
        "--powers",
        type=int,
        nargs="+",
        default=list(DEFAULT_POWERS),
        help="Use step sizes h = 2^(-k) for the provided powers k.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Random seeds for the vector field coefficients.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=MAX_DEGREE,
        help="Nilpotent truncation step.",
    )
    parser.add_argument(
        "--reference-refinement",
        type=int,
        default=REFERENCE_REFINEMENT,
        help="Refinement factor for the forward-order sanity check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for the saved convergence plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    powers = tuple(sorted(args.powers))
    hs = np.array([2.0 ** (-power) for power in powers], dtype=np.float64)
    seeds = list(args.seeds)
    backward_slopes: list[float] = []
    forward_slopes: list[float] = []
    plot_results: list[AggregatedResult] = []

    for rank in args.ranks:
        algebra = FreeNilpotentLyndon.build(rank, args.step)
        print(
            f"Built free nilpotent algebra rank={rank}, step={args.step}, dim={algebra.total_dimension}"
        )
        forward_y, backward_y = compute_error_curves(
            algebra, seeds, hs, args.reference_refinement
        )
        backward_floor = (
            10.0 * np.finfo(np.float64).eps * math.sqrt(algebra.total_dimension)
        )
        backward_fit_count = measurable_prefix_length(
            10.0**backward_y, absolute_floor=backward_floor
        )
        backward_slope = fitted_log_slope(
            hs[:backward_fit_count], backward_y[:backward_fit_count]
        )
        forward_slope = fitted_log_slope(hs[:-2], forward_y)
        print(f"rank={rank} averaged over seeds={seeds}")
        print(f"  fitted backward slope: {backward_slope:.4f}")
        print(f"  fitted forward slope:  {forward_slope:.4f}")
        backward_slopes.append(backward_slope)
        forward_slopes.append(forward_slope)
        plot_results.append(
            AggregatedResult(
                rank=rank,
                hs_forward=hs[:-2],
                forward_y=forward_y,
                hs_backward=hs,
                backward_y=backward_y,
                backward_fit_count=backward_fit_count,
            )
        )

    print("summary:")
    print(
        f"  backward slopes: min={min(backward_slopes):.4f}, "
        f"max={max(backward_slopes):.4f}, mean={np.mean(backward_slopes):.4f}"
    )
    print(
        f"  forward slopes:  min={min(forward_slopes):.4f}, "
        f"max={max(forward_slopes):.4f}, mean={np.mean(forward_slopes):.4f}"
    )
    output_path = save_plot_grid(plot_results, args.output_dir)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
