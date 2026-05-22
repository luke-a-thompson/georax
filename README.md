<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/georax/main/docs/_static/georax.dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/georax/main/docs/_static/georax.light.svg" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/luke-a-thompson/georax/main/docs/_static/georax.light.svg" width="350" alt="Logo">
  </picture>
</p>

<h2 align='center'>Geometric ODE and SDE solvers for Diffrax.</h2>

## Solvers

| Class | Problem type | Order | Automatic stepsizing | Notes |
|-------|--------------|-------|----------------------|-------|
| `GeometricEuler` | ODE/SDE | 1 ODE, 0.5 strong SDE | No | Euler and Euler-Maruyama update through a manifold chart |
| `RKMK(base_solver)` | ODE | base solver order | base solver dependent | Runge-Kutta-Munthe-Kaas lift of an explicit Diffrax RK solver |
| `SRKMK(base_solver)` | SDE | base solver dependent | base solver dependent | Stochastic Runge-Kutta-Munthe-Kaas lift of a Diffrax SRK solver |
| `CG2` | ODE/SDE | 2 | No | 2-stage Crouch-Grossman commutator-free method |
| `CG4` | ODE/SDE | 4 | No | 5-stage Crouch-Grossman commutator-free method |
| `CFEES25` | ODE/SDE | 2 | Yes | `CF-EES(2,5;1/10)`, O(1)-reversible, Stratonovich, 2N low-storage recurrence |
| `CFEES27` | ODE/SDE | 2 | No | `CF-EES(2,7;(5 - 3*sqrt(2))/14)`, O(1)-reversible, Stratonovich, 2N low-storage recurrence |

## Geometries

| Class | State | Coordinates | Chart |
|-------|-------|-------------|-------|
| `Euclidean()` | Any array | Same as state | Addition |
| `SO(n)` | `(n, n)` rotation matrix | `n * (n - 1) // 2` skew coordinates | Cayley at order 2, Taylor+QR at higher orders |
| `SPD(n)` | `(n, n)` symmetric positive-definite matrix | `n * (n + 1) // 2` symmetric coordinates | Congruence action via truncated exponential |

`GeometricTerm` is intrinsic: its vector field returns frame or Lie-algebra coordinates, not an ambient tangent matrix.

## Usage

```python
import diffrax
import jax.numpy as jnp
from georax import CFEES25, GeometricTerm, RKMK, SO


def coeffs(t, y, args):
    del t, y, args
    return jnp.array([-1.0, 0.0, 0.0])


term = GeometricTerm(coeffs, geometry=SO(3))
solvers = (RKMK(diffrax.Heun()), CFEES25())

for solver in solvers:
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=jnp.eye(3),
    )
```

For stochastic problems, use `GeometricEuler()` directly or wrap a Diffrax stochastic Runge-Kutta method with `SRKMK(...)`. Diffusion terms should return frame-coordinate coefficients compatible with the geometry.

`CFEES25` and `CFEES27` also work directly with SDE `MultiTerm`s. They are Stratonovich solvers, so the supplied drift and diffusion should describe the Stratonovich SDE in frame coordinates.

## Install

```bash
pip install georax
```

For development:

```bash
uv sync --extra dev
```

## Limitations

`RKMK` requires the selected chart to implement the inverse differential needed by the wrapped solver. The current `SO(n)` chart implements this for the order-2 Cayley chart, so `RKMK(diffrax.Heun())` is supported; higher-order adaptive `RKMK` on `SO(n)` needs additional chart support.

`CFEES25` and `CFEES27` are the commutator-free EES schemes from Shmelev, Thompson, and Salvi. They support both ODEs and SDEs, are O(1)-reversible, and converge to the Stratonovich solution for SDEs. The `CFEES25` coefficients correspond to `EES(2,5;1/10)`.

## Citation

If you use georax, please cite:

```bibtex
@article{ShmelevThompsonSalvi2025,
  title = {Explicit and Effectively Symmetric Schemes for Neural SDEs on Lie Groups},
  author = {Shmelev, Daniil and Thompson, Luke and Salvi, Cristopher},
  year = {2025},
  doi = {10.48550/arXiv.2509.20599},
  url = {https://arxiv.org/abs/2509.20599}
}
```

Core numerical references:

- Crouch and Grossman (1993), "Numerical integration of ordinary differential equations on manifolds", doi: `10.1007/BF02429858`.
- Munthe-Kaas (1998), "Runge-Kutta methods on Lie groups", doi: `10.1007/BF02510919`.
- Bazavov (2022), "Commutator-free Lie group methods with minimum storage requirements and reuse of exponentials", doi: `10.1007/s10543-021-00892-x`.

## Benchmarking

```bash
uv run pytest tests/benchmarks -m benchmark -k 'test_runtime or test_grad_runtime' -s
```
