import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Heun, ODETerm, SaveAt, diffeqsolve
from diffrax._custom_types import Args, RealScalarLike
from jaxtyping import Array

from georax import CFEES25, RKMK, SO, GeometricTerm


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


geo = SO(3)
inner = ODETerm(vf)
term = GeometricTerm(inner=inner, geometry=geo)
solvers = (
    ("CF-EES(2,5;1/4)", CFEES25()),
    ("RKMK(Heun())", RKMK(Heun())),
)

R0 = jnp.eye(3)
t0 = 0.0
t1 = 36.0
n_steps = 3600
ts = jnp.linspace(t0, t1, n_steps + 1)
dt0 = (t1 - t0) / n_steps

e1 = jnp.array([1.0, 0.0, 0.0])
trajectories = []
for name, solver in solvers:
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
    pts = jax.vmap(lambda R: R @ e1)(Rs)
    trajectories.append(
        (name, np.array(pts[:, 0]), np.array(pts[:, 1]), np.array(pts[:, 2]))
    )

t_color = np.array(ts)

fig = plt.figure(figsize=(12, 6))

u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones_like(u), np.cos(v))
for i, (name, x, y, z) in enumerate(trajectories, start=1):
    ax = fig.add_subplot(1, 2, i, projection="3d")  # type: ignore[assignment]
    ax.plot_wireframe(sx, sy, sz, color="lightgray", linewidth=0.4, alpha=0.4)
    ax.plot(x, y, z, color="black", linewidth=0.8, alpha=0.25)
    sc = ax.scatter(xs=x, ys=y, zs=z, c=t_color, cmap="viridis", s=2, linewidths=0)
    ax.scatter(x[0], y[0], z[0], color="crimson", s=40, label="start")
    ax.scatter(x[-1], y[-1], z[-1], color="goldenrod", s=40, label="end")
    ax.set_title(f"{name} on SO(3)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left")
    plt.colorbar(sc, ax=ax, label="t", shrink=0.6)

plt.tight_layout()
plt.savefig("lie_group_solvers_so3.png", dpi=150)
plt.show()
print("docs/examples/Saved lie_group_solvers_so3.png")
