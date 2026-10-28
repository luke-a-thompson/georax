"""Numerical checks for the independent high-precision documentation experiments."""

from decimal import Decimal, localcontext

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _convergence_plot import plot_curve, plt
from _decimal_cg4 import _refine, order_residuals
from _decimal_so3 import (
    deterministic_coeffs,
    identity,
    make_methods,
    norm,
    reference_solution,
    rotation,
    stochastic_coeffs,
)
from _problems import make_solver_accuracy_ambient_term, so3_body_frame_coeffs
from solver_convergence import solve_roundtrip
from stochastic_convergence import make_paths, solve_path

from georax import CFEES25, CFEES27, CG2, CG4, SO, CFCosineEES, GeometricTerm

jax.config.update("jax_enable_x64", True)
D = Decimal


@pytest.mark.parametrize("precision", [40, 80, 120])
def test_cg4_order_conditions_at_requested_precision(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        values = _refine(precision)
        assert max(map(abs, order_residuals(values))) < D(10) ** (2 - precision)


def test_cg4_resolves_fifth_order_local_errors_below_binary64():
    with localcontext() as ctx:
        ctx.prec = 80
        method = next(m for m in make_methods() if m.name == "CG4")
        errors = []
        for h in (D("0.0001"), D("0.00005")):
            y = method.step(
                D(0), h, identity(), lambda t, _, h=h: h * deterministic_coeffs(t)
            )
            errors.append(norm(y - reference_solution(h)))
        assert 31 < errors[0] / errors[1] < 33
        assert D("1e-30") < errors[-1] < D("1e-20")


@pytest.mark.parametrize(
    "index,solver",
    [(0, CG2()), (1, CG4()), (2, CFEES25()), (3, CFEES27()), (4, CFCosineEES(9))],
)
@pytest.mark.parametrize("stochastic", [False, True])
def test_decimal_stage_updates_match_georax(index, solver, stochastic):
    with localcontext() as ctx:
        ctx.prec = 60
        method = make_methods()[index]
        t, h = D("0.2"), D("0.125")
        y0 = rotation(np.array(list(map(D, ("0.2", "-0.1", "0.3")))), 8)
        dw = np.array(list(map(D, ("0.07", "-0.04"))))
        if stochastic:
            actual = method.step(t, h, y0, lambda _, y: stochastic_coeffs(y) @ dw)

            def diffusion(t, y, args):
                return jnp.array(
                    [
                        [-0.1 - 0.3 * y[2, 0], -0.8 - 0.15 * y[2, 2]],
                        [0.25 + 0.2 * y[1, 2], -0.35 + 0.2 * y[1, 1]],
                        [-0.9 - 0.2 * y[0, 0], -0.15 - 0.25 * y[0, 1]],
                    ]
                )

            term = diffrax.MultiTerm(
                GeometricTerm(lambda t, y, args: jnp.zeros(3), SO(3)),
                diffrax.ControlTerm(
                    diffusion, lambda t0, t1, **kwargs: jnp.array([0.07, -0.04])
                ),
            )
        else:
            actual = method.step(t, h, y0, lambda t, _: h * deterministic_coeffs(t))
            term = GeometricTerm(lambda t, y, args: so3_body_frame_coeffs(t), SO(3))
        y = jnp.asarray(y0.astype(float))
        t0, t1 = float(t), float(t + h)
        state = solver.init(term, t0, t1, y, None)
        expected = solver.step(term, t0, t1, y, None, state, False)[0]
        np.testing.assert_allclose(actual.astype(float), expected, rtol=0, atol=3e-14)


def test_decimal_reference_precision_and_independent_ode_solution():
    with localcontext() as ctx:
        ctx.prec = 50
        low = reference_solution()
        ctx.prec = 80
        high = reference_solution()
        assert norm(low - high) < D("1e-49")
        assert norm(high.T @ high - identity()) < D("1e-78")
    solution = diffrax.diffeqsolve(
        make_solver_accuracy_ambient_term(),
        diffrax.Dopri8(),
        0,
        1,
        0.01,
        jnp.eye(3),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-12),
    )
    np.testing.assert_allclose(high.astype(float), solution.ys[0], rtol=0, atol=2e-12)


@pytest.mark.parametrize(
    "name,order", [("CF-EES25", 5), ("CF-EES27", 7), ("CF-EES29", 9)]
)
def test_decimal_roundtrip_orders_below_binary64(name, order):
    with localcontext() as ctx:
        ctx.prec = 60
        method = next(m for m in make_methods() if m.name == name)
        errors = [norm(solve_roundtrip(method, n)[1] - identity()) for n in (128, 256)]
        measured = (errors[0] / errors[1]).ln() / D(2).ln()
        assert abs(measured - order) < D("0.02")
        assert errors[-1] < D("1e-12")


@pytest.mark.parametrize("hurst", [D("0.4"), D("0.5"), D("0.6")])
def test_decimal_paths_are_reproducible_and_nested(hurst):
    with localcontext() as ctx:
        ctx.prec = 60
        first = make_paths(2, 5, 123, hurst)
        repeat = make_paths(2, 5, 123, hurst)
        for x, y in zip(first, repeat, strict=True):
            np.testing.assert_array_equal(x, y)
            assert all(isinstance(value, Decimal) for value in x.flat)
            # Coarse increments are sums of fine increments, to Decimal roundoff.
            coarse = np.diff(x[::4], axis=0)
            fine = np.diff(x, axis=0).reshape(8, 4, 2).sum(axis=1)
            assert norm(coarse - fine) < D("1e-57")


def test_zero_control_and_roundoff_mask():
    with localcontext() as ctx:
        ctx.prec = 40
        path = np.full((3, 2), D(0), dtype=object)
        for method in make_methods():
            np.testing.assert_array_equal(solve_path(method, path)[-1], identity())
        fig, ax = plt.subplots()
        result = plot_curve(
            "test",
            [D("0.5"), D("0.25"), D("0.125")],
            [D("0.01"), D("0.0025"), D(0)],
            2,
            40,
            ax,
        )
        assert result["resolved"] == [True, True, False]
        assert result["measured_slope"] == pytest.approx(2)
        plt.close(fig)
