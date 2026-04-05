from __future__ import annotations

"""SPD diffusion example integrated with CF-EES25 and Diffrax."""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from diffrax import (
    CheckpointedReversibleAdjoint,
    ControlTerm,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)

from georax import CFEES25, SPD, GeometricTerm

matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="`CFEES25` is not marked as converging to either the Itô or the Stratonovich solution.",
)


def sym(matrix):
    return 0.5 * (matrix + matrix.T)


def spd_sqrt(matrix):
    matrix = sym(matrix)
    evals, evecs = jnp.linalg.eigh(matrix)
    evals = jnp.clip(evals, min=jnp.finfo(matrix.dtype).eps)
    return (evecs * jnp.sqrt(evals)) @ evecs.T


def spd_invsqrt(matrix):
    matrix = sym(matrix)
    evals, evecs = jnp.linalg.eigh(matrix)
    evals = jnp.clip(evals, min=jnp.finfo(matrix.dtype).eps)
    return (evecs * jnp.reciprocal(jnp.sqrt(evals))) @ evecs.T


def spd_log(matrix):
    matrix = sym(matrix)
    evals, evecs = jnp.linalg.eigh(matrix)
    evals = jnp.clip(evals, min=jnp.finfo(matrix.dtype).eps)
    return (evecs * jnp.log(evals)) @ evecs.T


def airm_log(x, y):
    sqrt_x = spd_sqrt(x)
    invsqrt_x = spd_invsqrt(x)
    mid = sym(invsqrt_x @ y @ invsqrt_x)
    return sym(sqrt_x @ spd_log(mid) @ sqrt_x)


def diffusion_columns(x, basis, sigma):
    sqrt_x = spd_sqrt(x)
    return sigma * jnp.einsum("ab,bck,cd->adk", sqrt_x, basis, sqrt_x)


key = jax.random.key(0)
geometry = SPD(3)
solver = CFEES25()

t0 = 0.0
t1 = 8.0
num_steps = 400
dt0 = (t1 - t0) / num_steps
ts = jnp.linspace(t0, t1, num_steps + 1)

x0 = jnp.array(
    [
        [1.4, 0.1, 0.0],
        [0.1, 1.0, 0.1],
        [0.0, 0.1, 0.8],
    ]
)
target = jnp.array(
    [
        [2.2, 0.3, 0.1],
        [0.3, 1.6, 0.2],
        [0.1, 0.2, 1.1],
    ]
)
kappa = 0.45
sigma = 0.18
basis = geometry.frame(jnp.eye(geometry.n, dtype=x0.dtype))
brownian_path = VirtualBrownianTree(
    t0=t0,
    t1=t1,
    tol=dt0 / 4.0,
    shape=(geometry.dimension,),
    key=key,
)


def drift(t, x, args):
    del t, args
    return kappa * airm_log(x, target)


def diffusion(t, x, args):
    del t, args
    return diffusion_columns(x, basis, sigma)


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
    y0=x0,
    saveat=SaveAt(ts=ts),
    adjoint=CheckpointedReversibleAdjoint(checkpoint_every=32),
    max_steps=num_steps + 10,
)

xs = solution.ys
eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(xs)

fig, ax = plt.subplots(figsize=(8, 4.5))
for i in range(3):
    ax.plot(
        np.array(ts),
        np.array(eigenvalues[:, i]),
        linewidth=2,
        label=f"$\\lambda_{i + 1}$",
    )
ax.set_title("SPD(3) diffusion under AIRM retraction with CF-EES25")
ax.set_xlabel("t")
ax.set_ylabel("eigenvalue")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/examples/outputs/spd_sde_cfees25_eigenvalues.png", dpi=500)
print("Saved spd_sde_cfees25_eigenvalues.png")
