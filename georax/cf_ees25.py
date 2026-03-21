from __future__ import annotations

from typing import Any
from typing import cast, override

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diffrax import ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from diffrax._solution import RESULTS

from ._geometry import SO3Ops
from ._term import GeometricTerm
from .base import AbstractCommutatorFreeSolver


class CFEES25(AbstractCommutatorFreeSolver):
    """Commutator-free EES(2,5;1/4) solver with chained exponentials."""

    @override
    def step(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, None, DenseInfo, None, RESULTS]:
        del solver_state, made_jump

        dt = t1 - t0
        geometric_term = self._unwrap_geometric_term(terms)
        y0_array = cast(Array, y0)
        control = terms.contr(t0, t1)

        # Use midpoint/end-point evaluations for the nonautonomous extension.
        k1 = geometric_term.coeffs(t0, y0_array, args)
        delta_y1 = jnp.asarray(control, dtype=k1.dtype) * k1
        y1 = geometric_term.frozen_flow(y0_array, cast(Array, 0.5 * delta_y1))

        t_mid = t0 + 0.5 * dt
        k2 = geometric_term.coeffs(t_mid, y1, args)
        delta_y2 = cast(
            Array, -0.5 * delta_y1 + jnp.asarray(control, dtype=k2.dtype) * k2
        )
        y2 = geometric_term.frozen_flow(y1, delta_y2)

        k3 = geometric_term.coeffs(t1, y2, args)
        delta_y3 = cast(
            Array, -2.0 * delta_y2 + jnp.asarray(control, dtype=k3.dtype) * k3
        )
        y3 = geometric_term.frozen_flow(y2, cast(Array, 0.25 * delta_y3))

        dense_info = dict(y0=y0, y1=y3)
        return y3, None, dense_info, None, RESULTS.successful

    @override
    def order(self, terms: GeometricTerm) -> int | None:
        del terms
        return 2

    @override
    def func(
        self,
        terms: GeometricTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def omega_body(t: RealScalarLike) -> Array:
        t_array = jnp.asarray(t)
        return jnp.array(
            [
                0.8 + 0.45 * jnp.sin(0.7 * t_array),
                0.55 * jnp.cos(1.3 * t_array + 0.2),
                0.35 + 0.6 * jnp.sin(0.9 * t_array - 0.4),
            ]
        )

    def vf(t: RealScalarLike, R: Array, args: Args) -> Array:
        del args
        omega = omega_body(t)
        skew = jnp.array(
            [
                [0.0, -omega[2], omega[1]],
                [omega[2], 0.0, -omega[0]],
                [-omega[1], omega[0], 0.0],
            ]
        )
        return R @ skew

    geo = SO3Ops()
    inner = ODETerm(vf)
    term = GeometricTerm(inner=inner, geometry=geo)
    solver = CFEES25()

    R0 = jnp.eye(3)
    t0 = 0.0
    t1 = 36.0
    n_steps = 3600
    ts = jnp.linspace(t0, t1, n_steps + 1)
    dt0 = (t1 - t0) / n_steps

    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=R0,
        saveat=SaveAt(ts=ts),
    )
    Rs = sol.ys

    e1 = jnp.array([1.0, 0.0, 0.0])
    pts = jax.vmap(lambda R: R @ e1)(Rs)
    x = np.array(pts[:, 0])
    y = np.array(pts[:, 1])
    z = np.array(pts[:, 2])
    t_color = np.array(ts)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color="lightgray", linewidth=0.4, alpha=0.4)

    ax_3d = cast(Any, ax)
    ax_3d.plot(x, y, z, color="black", linewidth=0.8, alpha=0.25)
    sc = ax_3d.scatter(xs=x, ys=y, zs=z, c=t_color, cmap="viridis", s=2, linewidths=0)
    ax_3d.scatter(x[0], y[0], z[0], color="crimson", s=40, label="start")
    ax_3d.scatter(x[-1], y[-1], z[-1], color="goldenrod", s=40, label="end")
    plt.colorbar(sc, ax=ax, label="t", shrink=0.6)

    ax.set_title("CF-EES(2,5;1/4) on SO(3) - swept trajectory of $R(t) e_1$ on $S^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("cf_ees25_so3.png", dpi=150)
    plt.show()
    print("Saved cf_ees25_so3.png")
