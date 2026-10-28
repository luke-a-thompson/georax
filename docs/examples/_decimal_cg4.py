"""Refine the five-stage CG4 tableau using its 13 order conditions.

The starting values are the Jackiewicz--Marthinsen--Owren tableau reproduced in
Andrle and Crassidis, Table 5:
https://ancs.eng.buffalo.edu/pdf/ancs_papers/2012/geom_int.pdf
The Crouch--Grossman order theory is due to Owren and Marthinsen:
https://doi.org/10.1023/A:1022325426017

There are two free parameters. We fix a21 and a31 at the published decimal
values and solve for the other 13 coefficients. This defines a nearby tableau
at arbitrary precision, rather than claiming extra published digits. All eight
classical RK conditions and all five additional CG conditions are enforced.
"""

from __future__ import annotations

from decimal import Decimal, getcontext, localcontext
from functools import lru_cache

D = Decimal

_SEED = (
    "0.8177227988124852",
    "0.3199876375476427",
    "0.0659864263556022",
    "0.9214417194464946",
    "0.4997857776773573",
    "-1.0969984448371582",
    "0.3552358559023322",
    "0.2390958372307326",
    "1.3918565724203246",
    "-1.1092979392113565",
    "0.1370831520630755",
    "-0.0183698531564020",
    "0.7397813985370780",
    "-0.1907142565505889",
    "0.3322195591068374",
)


def unpack(
    values: tuple[Decimal, ...] | list[Decimal],
) -> tuple[tuple[tuple[Decimal, ...], ...], tuple[Decimal, ...]]:
    rows = tuple(tuple(values[i * (i - 1) // 2 : i * (i + 1) // 2]) for i in range(5))
    return rows, tuple(values[10:])


def order_residuals(values: tuple[Decimal, ...] | list[Decimal]) -> list[Decimal]:
    """Eight classical plus five CG conditions, for products in stage order.

    S(w, v) is the coefficient of F P in a product of exponentials with
    increments h*w_i*(F + h*v_i*P). T(w, v) is its F F P coefficient.
    The extra conditions match the independent ordered terms F P, D(F P),
    F D(F^2), F D(P), and F F P, with P = D(F) and D = f'(y).
    """
    a, b = unpack(values)
    c = [sum(row, D(0)) for row in a]

    def dot(u, v):
        return sum((x * y for x, y in zip(u, v, strict=True)), D(0))

    def matvec(v):
        return [dot(row, v[: len(row)]) for row in a]

    def s(w, v):
        prefix = D(0)
        result = D(0)
        for weight, value in zip(w, v, strict=True):
            result += weight * value * (prefix + weight / 2)
            prefix += weight
        return result

    ac = matvec(c)
    c2 = [x * x for x in c]
    t = D(0)
    prefix = D(0)
    for weight, value in zip(b, c, strict=True):
        t += (
            weight
            * value
            * (prefix * prefix / 2 + prefix * weight / 2 + weight * weight / 6)
        )
        prefix += weight
    return [
        sum(b) - 1,
        dot(b, c) - D(1) / 2,
        dot(b, c2) - D(1) / 3,
        dot(b, ac) - D(1) / 6,
        dot(b, [x**3 for x in c]) - D(1) / 4,
        dot(b, [x * y for x, y in zip(c, ac, strict=True)]) - D(1) / 8,
        dot(b, matvec(c2)) - D(1) / 12,
        dot(b, matvec(ac)) - D(1) / 24,
        s(b, c) - D(1) / 3,
        dot(b, [s(row, c[: len(row)]) for row in a]) - D(1) / 12,
        s(b, c2) - D(1) / 4,
        s(b, ac) - D(1) / 8,
        t - D(1) / 8,
    ]


def _linear_solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
    augmented = [row + [value] for row, value in zip(matrix, rhs, strict=True)]
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(augmented[i][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        if not augmented[col][col]:
            raise ArithmeticError("Singular CG4 order-condition Jacobian.")
        for i in range(col + 1, n):
            scale = augmented[i][col] / augmented[col][col]
            for j in range(col + 1, n + 1):
                augmented[i][j] -= scale * augmented[col][j]
    solution = [D(0)] * n
    for i in range(n - 1, -1, -1):
        solution[i] = (
            augmented[i][-1]
            - sum((augmented[i][j] * solution[j] for j in range(i + 1, n)), D(0))
        ) / augmented[i][i]
    return solution


@lru_cache(maxsize=8)
def _refine(precision: int) -> tuple[Decimal, ...]:
    with localcontext() as ctx:
        # Central differences with this step and guard precision retain more
        # than precision digits in the Jacobian, whose entries are polynomials.
        ctx.prec = 3 * precision + 30
        values = list(map(D, _SEED))
        delta = D(10) ** (-precision - 5)
        tolerance = D(10) ** (-precision - 10)
        for _ in range(12):
            residual = order_residuals(values)
            if max(map(abs, residual)) < tolerance:
                break
            columns = []
            for j in range(2, 15):
                plus, minus = values.copy(), values.copy()
                plus[j] += delta
                minus[j] -= delta
                columns.append(
                    [
                        (p - m) / (2 * delta)
                        for p, m in zip(
                            order_residuals(plus), order_residuals(minus), strict=True
                        )
                    ]
                )
            update = _linear_solve(
                [list(row) for row in zip(*columns, strict=True)], residual
            )
            values[2:] = [x - dx for x, dx in zip(values[2:], update, strict=True)]
        else:
            raise ArithmeticError("CG4 coefficient refinement did not converge.")
        ctx.prec = precision
        return tuple(+x for x in values)


def make_cg4_coefficients() -> tuple[
    tuple[Decimal, ...], tuple[tuple[Decimal, ...], ...], tuple[Decimal, ...]
]:
    a, b = unpack(_refine(getcontext().prec))
    c = tuple(sum(row, D(0)) for row in a)
    return c, a, b
