"""Group Brownian motion on the sphere via stochastic rotations in SO(3)."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from diffrax import (
    ControlTerm,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)

from georax import CFEES25, SO, GeometricTerm

matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="`CFEES25` is not marked as converging to either the Itô or the Stratonovich solution.",
)


def drift_omega(t):
    t = jnp.asarray(t)
    return jnp.array(
        [
            0.3 + 0.1 * jnp.sin(0.5 * t),
            -0.2 + 0.15 * jnp.cos(0.7 * t),
            0.4,
        ]
    )


geometry = SO(3)
solver = CFEES25()
key = jax.random.key(0)

t0 = 0.0
t1 = 15.0
num_steps = 2_000
dt0 = (t1 - t0) / num_steps
ts = jnp.linspace(t0, t1, num_steps + 1)

R0 = jnp.eye(3)
e3 = jnp.array([0.0, 0.0, 1.0])
sigma = 0.35
brownian_path = VirtualBrownianTree(
    t0=t0,
    t1=t1,
    tol=dt0 / 4.0,
    shape=(3,),
    key=key,
)


def drift(t, R, args):
    del args
    return geometry.from_frame(R, drift_omega(t))


def diffusion(t, R, args):
    del t, args
    return sigma * geometry.frame(R)


term = GeometricTerm(
    inner=MultiTerm(
        ODETerm(drift),
        ControlTerm(diffusion, brownian_path),
    ),
    geometry=geometry,
)
solution = diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=R0,
    saveat=SaveAt(ts=ts),
    max_steps=num_steps + 10,
)

Rs = solution.ys
pts = jax.vmap(lambda R: R @ e3)(Rs)
grams = jax.vmap(lambda R: R.T @ R)(Rs)
dets = jax.vmap(jnp.linalg.det)(Rs)
radii = jnp.linalg.norm(pts, axis=1)
orth_err = jnp.max(jnp.abs(grams - jnp.eye(3)))
radial_err = jnp.max(jnp.abs(radii - 1.0))

assert float(orth_err) < 3e-3
assert float(jnp.min(dets)) > 0.999
assert float(radial_err) < 1e-3

x = np.array(pts[:, 0])
y = np.array(pts[:, 1])
z = np.array(pts[:, 2])
t_color = np.array(ts)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")  # type: ignore[assignment]

u = np.linspace(0.0, 2.0 * np.pi, 50)
v = np.linspace(0.0, np.pi, 25)
sx = np.outer(np.cos(u), np.sin(v))
sy = np.outer(np.sin(u), np.sin(v))
sz = np.outer(np.ones_like(u), np.cos(v))

ax.plot_wireframe(sx, sy, sz, color="lightgray", linewidth=0.4, alpha=0.35)
ax.plot(x, y, z, color="black", linewidth=0.8, alpha=0.25)
sc = ax.scatter(xs=x, ys=y, zs=z, c=t_color, cmap="viridis", s=3, linewidths=0)
ax.scatter(x[0], y[0], z[0], color="crimson", s=45, label="start")
ax.scatter(x[-1], y[-1], z[-1], color="goldenrod", s=45, label="end")
ax.set_title("Group Brownian Motion on $S^2$ via SO(3)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_box_aspect([1, 1, 1])
ax.legend(loc="upper left")
plt.colorbar(sc, ax=ax, label="t", shrink=0.7)

plt.tight_layout()
plt.savefig("docs/examples/outputs/sphere_gbm_so3.png", dpi=500)
print("Saved sphere_gbm_so3.png")
