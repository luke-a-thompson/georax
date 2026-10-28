"""Decimal versions of the SO(3) convergence problems and solver stage updates.

These are documentation experiments, not a Decimal backend for Diffrax. NumPy
object arrays store Decimal entries throughout; no state arithmetic uses floats.
The charts match Georax: Cayley for CG2, and Taylor + positive-diagonal QR for
CG4 and CF-EES (degrees 4, 8, 8, and 12 respectively).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, getcontext, localcontext
from math import factorial

import numpy as np
from _decimal_cg4 import make_cg4_coefficients
from numpy.typing import NDArray

D = Decimal
Array = NDArray[np.object_]
VectorField = Callable[[Decimal, Array], Array]


def identity() -> Array:
    return np.array([[D(int(i == j)) for j in range(3)] for i in range(3)])


def norm(x: Array) -> Decimal:
    return sum((v * v for v in x.flat), D(0)).sqrt()


def skew(a: Array) -> Array:
    """Georax's upper-triangular coordinates (01, 02, 12)."""
    return np.array([[D(0), a[0], a[1]], [-a[0], D(0), a[2]], [-a[1], -a[2], D(0)]])


def rotation(a: Array, order: int) -> Array:
    omega = skew(a)
    r2 = sum((v * v for v in a), D(0))
    if order <= 2:
        return identity() + (omega + (omega @ omega) / 2) / (1 + r2 / 4)

    if not r2:
        return identity()
    # Georax rounds up to its optimized degree-4, degree-8, and degree-12
    # Taylor polynomials before falling back to the requested even degree.
    degree = next((d for d in (4, 8, 12) if d >= order), order + order % 2)
    # For a 3x3 skew matrix, omega**3 = -r2 * omega. Evaluate the same
    # truncated Taylor polynomial as Georax without large matrix powers.
    u = sum(((-r2) ** k / D(factorial(2 * k + 1)) for k in range(degree // 2)), D(0))
    v = sum(((-r2) ** k / D(factorial(2 * k + 2)) for k in range(degree // 2)), D(0))
    polynomial = identity() + u * omega + v * (omega @ omega)
    columns = []
    for column in polynomial.T:
        q = column.copy()
        for previous in columns:
            q -= (previous @ q) * previous
        columns.append(q / norm(q))
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class DecimalMethod:
    name: str
    title: str
    order: int
    C: tuple[Decimal, ...]
    A: tuple[Decimal, ...] = ()
    B: tuple[Decimal, ...] = ()
    stage_weights: tuple[tuple[Decimal, ...], ...] = ()
    final_weights: tuple[Decimal, ...] = ()
    antisymmetric_order: int | None = None

    @property
    def chart_order(self) -> int:
        return self.antisymmetric_order or self.order

    def step(self, t: Decimal, h: Decimal, y: Array, vf: VectorField) -> Array:
        """Advance once; vf returns the full step's frame-coordinate increment."""
        if self.B:
            tmp = np.array([D(0)] * 3)
            for i, (b, c) in enumerate(zip(self.B, self.C, strict=True)):
                increment = vf(t + c * h, y)
                tmp = self.A[i - 1] * tmp + increment if i else increment
                y = y @ rotation(b * tmp, self.chart_order)
            return y

        stages = []
        for c, weights in zip(self.C, self.stage_weights, strict=True):
            stage_y = y
            for weight, stage in zip(weights, stages, strict=True):
                if weight:
                    stage_y = stage_y @ rotation(weight * stage, self.chart_order)
            stages.append(vf(t + c * h, stage_y))
        for weight, stage in zip(self.final_weights, stages, strict=True):
            if weight:
                y = y @ rotation(weight * stage, self.chart_order)
        return y

    def metadata(self) -> dict[str, object]:
        return {
            "order": self.order,
            "antisymmetric_order": self.antisymmetric_order,
            "chart": "Cayley" if self.chart_order <= 2 else "Taylor + QR",
            "chart_order": self.chart_order,
            "coefficients": {
                "A": [str(x) for x in self.A],
                "B": [str(x) for x in self.B],
                "C": [str(x) for x in self.C],
                "stage_weights": [[str(x) for x in row] for row in self.stage_weights],
                "final_weights": [str(x) for x in self.final_weights],
            },
        }


def make_methods() -> tuple[DecimalMethod, ...]:
    """Recompute rational and radical coefficients in the active context."""
    s2 = D(2).sqrt()
    s5 = D(5).sqrt()
    cg4_c, cg4_a, cg4_b = make_cg4_coefficients()
    return (
        DecimalMethod(
            "CG2",
            r"\mathrm{CG2}",
            2,
            (D(0), D(1) / 2),
            stage_weights=((), (D(1) / 2,)),
            final_weights=(D(0), D(1)),
        ),
        DecimalMethod(
            "CG4",
            r"\mathrm{CG4}",
            4,
            cg4_c,
            stage_weights=cg4_a,
            final_weights=cg4_b,
        ),
        DecimalMethod(
            "CF-EES25",
            r"\mathrm{CF\text{-}EES}_{\mathcal{R}}(2,5)",
            2,
            (D(0), D(1) / 3, D(5) / 6),
            A=(-D(7) / 15, -D(35) / 32),
            B=(D(1) / 3, D(15) / 16, D(2) / 5),
            antisymmetric_order=5,
        ),
        DecimalMethod(
            "CF-EES27",
            r"\mathrm{CF\text{-}EES}_{\mathcal{R}}(2,7)",
            2,
            (D(0), (2 - s2) / 3, (2 + s2) / 6, (4 + s2) / 6),
            A=((-7 + 4 * s2) / 3, -(4 + 5 * s2) / 12, 3 * (-31 + 8 * s2) / 49),
            B=((2 - s2) / 3, (4 + s2) / 8, 3 * (3 - s2) / 7, (9 - 4 * s2) / 14),
            antisymmetric_order=7,
        ),
        DecimalMethod(
            "CF-EES29",
            r"\mathrm{CF\text{-}EES}_{\mathcal{R}}(2,9)",
            2,
            (D(0), (3 - s5) / 4, D(1) / 2, (1 + s5) / 4, D(1)),
            A=((s5 - 3) / 2, -(1 + s5) / 4, 1 - s5, -(3 + s5) / 2),
            B=((3 - s5) / 4, D(1) / 2, (s5 - 1) / 2, D(1) / 2, (3 - s5) / 8),
            antisymmetric_order=9,
        ),
    )


def sincos(x: Decimal) -> tuple[Decimal, Decimal]:
    """Taylor evaluation for the small arguments in the deterministic problem."""
    with localcontext() as ctx:
        ctx.prec += 8
        s, c = x, D(1)
        st, ct = s, c
        for k in range(1, 4 * ctx.prec):
            st *= -x * x / ((2 * k) * (2 * k + 1))
            ct *= -x * x / ((2 * k - 1) * (2 * k))
            old = s, c
            s, c = s + st, c + ct
            if (s, c) == old:
                break
        else:
            raise ArithmeticError("Decimal sine/cosine series did not converge.")
    return +s, +c


def deterministic_coeffs(t: Decimal) -> Array:
    """The same time-dependent field as _problems.so3_body_frame_coeffs."""
    wx = D("0.8") + D("0.45") * sincos(D("0.7") * t)[0]
    wy = D("0.55") * sincos(D("1.3") * t + D("0.2"))[1]
    wz = D("0.35") + D("0.6") * sincos(D("0.9") * t - D("0.4"))[0]
    return np.array([-wz, wy, -wx])


def reference_solution(t: Decimal = Decimal(1)) -> Array:
    """Solve R' = R Omega(t) by an independent Decimal power series.

    If Omega(t) = sum A_k t^k, then R_{n+1} = sum R_j A_{n-j} / (n+1).
    Evaluate with ten guard digits, stopping after twelve consecutive tiny terms.
    This avoids a binary64 reference floor in the forward-error curves.
    """
    precision = getcontext().prec
    with localcontext() as ctx:
        ctx.prec += 10
        frequencies = tuple(map(D, ("0.7", "1.3", "0.9")))
        phases = (sincos(D(0)), sincos(D("0.2")), sincos(D("-0.4")))
        s = [p[0] for p in phases]
        c = [p[1] for p in phases]
        generators = []
        coefficients = [identity()]
        result = identity()
        small_terms = 0
        for n in range(8 * precision + 64):
            omega = [D("0.45") * s[0], D("0.55") * c[1], D("0.6") * s[2]]
            if n == 0:
                omega[0] += D("0.8")
                omega[2] += D("0.35")
            generators.append(skew(np.array([-omega[2], omega[1], -omega[0]])))
            coefficient = sum(
                (coefficients[j] @ generators[n - j] for j in range(n + 1)),
                identity() * 0,
            ) / (n + 1)
            coefficients.append(coefficient)
            term = coefficient * t ** (n + 1)
            result += term
            small_terms = (
                small_terms + 1 if norm(term) < D(10) ** (-precision - 5) else 0
            )
            if small_terms == 12:
                break
            s, c = (
                [frequencies[i] * c[i] / (n + 1) for i in range(3)],
                [-frequencies[i] * s[i] / (n + 1) for i in range(3)],
            )
        else:
            raise ArithmeticError("Decimal reference series did not converge.")
    return np.array([[+x for x in row] for row in result])


def stochastic_coeffs(y: Array) -> Array:
    """The original state-dependent, two-channel SO(3) diffusion, sigma=1."""
    wx = [D("0.9") + D("0.2") * y[0, 0], D("0.15") + D("0.25") * y[0, 1]]
    wy = [D("0.25") + D("0.2") * y[1, 2], -D("0.35") + D("0.2") * y[1, 1]]
    wz = [D("0.1") + D("0.3") * y[2, 0], D("0.8") + D("0.15") * y[2, 2]]
    return np.array([[-v for v in wz], wy, [-v for v in wx]])
