# georax

Geometric ODE solvers for [Diffrax](https://github.com/patrick-kidger/diffrax), focused on Lie-group and manifold integration.

## RKMK

`RKMK` wraps an explicit Diffrax Runge-Kutta solver and lifts it to geometric integration on a `GeometricTerm`.
It reuses the wrapped solver's Runge-Kutta coefficients, so you choose accuracy/adaptivity by choosing the base solver.

- Wrap a fixed-step ERK (for example `Heun()`) for fixed-step geometric integration.
- Wrap an adaptive ERK (for example `Dopri5()`) to keep automatic stepsizing.

## Commutator-Free Solvers

| Class | Stages | Order | Automatic stepsizing | Notes |
|-------|--------|-------|----------------------|-------|
| `CG2` | 2 | 2 | No | - |
| `CG4` | 5 | 4 | No | - |
| `CFEES25` | 3 | 2 | No | 2N low-storage recurrence |

## Usage

```python
import diffrax
import jax.numpy as jnp
from georax import CFEES25, GeometricTerm, RKMK, SO

def vf(t, y, args):
    del t, args
    return y @ jnp.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

term = GeometricTerm(inner=diffrax.ODETerm(vf), geometry=SO(3))

# Wrap a base Diffrax solver to work on a Lie group, or use a specialized manifold solver.
for solver in (RKMK(diffrax.Dopri5()), CFEES25()):
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0, t1=1.0, dt0=0.01, y0=jnp.eye(3),
    )
```

## Install

```bash
uv sync
```
