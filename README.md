<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/georax/main/docs/_static/georax.dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/georax/main/docs/_static/georax.light.svg" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/luke-a-thompson/georax/main/docs/_static/georax.light.svg" width="350" alt="Logo">
  </picture>
</p>

<h2 align='center'>Geometric ODE and SDE solvers for Diffrax.</h2>

Georax provides geometric numerical integrators for the [Diffrax](https://github.com/patrick-kidger/diffrax) ecosystem. Georax supports solving ODEs, SDEs and CDEs over homogeneous spaces and enables efficient training of intrinsic neural differential equations with $\mathcal{O}(1)$ backpropagation via EES methods. RDE support is provided by integration with [Roughrax](https://github.com/luke-a-thompson/roughrax).

## Solvers

| Class | Problem type | Order | Automatic stepsizing | Notes |
|-------|--------------|-------|----------------------|-------|
| `GeometricEuler` | ODE/SDE | 1 ODE, 0.5 strong SDE | No | Euler–Maruyama in local chart coordinates |
| `GeometricEulerHeun` | SDE | 1 ODE, 0.5 strong SDE | No | Stratonovich; Euler drift and Heun diffusion |
| `RKMK(base_solver)` | ODE | base solver order | base solver dependent | Runge-Kutta-Munthe-Kaas lift of an explicit Diffrax RK solver |
| `SRKMK(base_solver)` | SDE | base solver dependent | base solver dependent | Stochastic Runge-Kutta-Munthe-Kaas lift of a Diffrax SRK solver |
| `CG2` | ODE/SDE | 2 | No | 2-stage Crouch-Grossman commutator-free method |
| `CG4` | ODE/SDE | 4 | No | 5-stage Crouch-Grossman commutator-free method |
| `CFEES25` | ODE/SDE | 2 | Yes | `CF-EES(2,5;1/10)`, O(1)-reversible, Stratonovich, 2N low-storage recurrence |
| `CFEES27` | ODE/SDE | 2 | Yes | `CF-EES(2,7;(5 - 3*sqrt(2))/14)`, O(1)-reversible, Stratonovich, 2N low-storage recurrence |
| `CFCosineEES(p)` | ODE/SDE | 2 | No | Cosine EES with odd reversible order `p >= 3`, `(p + 1) // 2` stages, Stratonovich, 2N low-storage recurrence |

## Geometries

| Class | State | Coordinates | Chart |
|-------|-------|-------------|-------|
| `Euclidean()` | Any array | Same as state | Addition |
| `Euclidean(shape)` | Array with the specified shape | Same as state | Addition |
| `Torus(d)` | `(d,)` angles in `[-pi, pi)` | `(d,)` angular increments | Exact wrapped addition |
| `TangentTorus(n)` | `(2 * n,)`, packed as `[theta, omega]` | `(2 * n,)` increments | Product of wrapped and ordinary addition |
| `Sphere(n)` | `(..., n)` unit vectors in R^n | `(..., n * (n - 1) // 2)` skew coordinates | SO(n) rotation action, with Cayley or Taylor+QR |
| `SO(n)` | `(n, n)` rotation matrix | `n * (n - 1) // 2` skew coordinates | Cayley at order 2, Taylor+QR at higher orders |
| `SPD(n)` | `(n, n)` symmetric positive-definite matrix | `n * (n + 1) // 2` symmetric coordinates | Congruence action via truncated exponential |
| `Product(*factors)` | Flat concatenation of factor states | Flat concatenation of factor coordinates | Componentwise factor charts |

`GeometricTerm` is intrinsic: its vector field returns frame or Lie-algebra coordinates, not an ambient tangent matrix.

Direct geometry calls take an explicit chart: `geometry.apply_increment(x, a, geometry.select_chart(order))`. Chart selection does not modify the geometry.

`Sphere(n)` denotes the sphere of dimension `n - 1`. Its skew coordinates fill the strictly lower triangle in row order: `(1, 0), (2, 0), (2, 1), (3, 0), ...`, with the negative transpose in the upper triangle. The infinitesimal action is `A @ x`. This differs from `SO(n)`, which retains its upper-triangular coordinate convention and action `x @ A`. Sphere generators are redundant: `trivialise(x, v)` chooses the lift `v x.T - x v.T`, and `detrivialise(x, trivialise(x, v)) == v` for tangent vectors at unit states. A sphere accepts leading batch dimensions and either shared or per-state rotation coefficients.

### Product geometries

`Product(SPD(2), Torus(3), SO(3))` composes the existing geometries without changing the solvers. Use `pack_state` and `unpack_state` to move between the flat solver state and the factor arrays, and `pack_coordinates` and `unpack_coordinates` for vector-field coefficients. State and coordinate lengths may differ. Factors must have fixed shapes, so use `Euclidean(n)` or `Euclidean((m, n))` inside a product. Products can be nested; `unpack_state(sol.ys)` retains the leading observation axis. Use `jax.vmap` to batch complete product solves.

`TangentTorus(n)` is a convenience subclass of `Product(Torus(n), Euclidean(n))`, with no separate geometric implementation. A model can couple the factors while the geometry handles their charts:

```python
import jax.numpy as jnp

from georax import GeometricTerm, TangentTorus

geometry = TangentTorus(3)
y0 = geometry.pack_state(jnp.zeros(3), jnp.ones(3))


def rotor_coeffs(t, y, damping):
    theta, omega = geometry.unpack_state(y)
    return geometry.pack_coordinates(omega, -jnp.sin(theta) - damping * omega)


term = GeometricTerm(rotor_coeffs, geometry)
```

Only the phase components are wrapped. Torus charts use exact addition modulo `2 * pi` at every requested order. Derivatives of angular representatives are local to the chosen branch; periodic observables such as sine and cosine avoid branch discontinuities. Product charts preserve each factor's pullback selection and keep all chart orders static for reversible adjoints.

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

For SDEs, use `GeometricEuler()` for Itô in local chart coordinates, `GeometricEulerHeun()` for Stratonovich, or `SRKMK(...)` to wrap a Diffrax stochastic Runge-Kutta method. Supply frame-coordinate coefficients; nonlinear charts contribute an additional ambient Itô drift. `GeometricEulerHeun()` expects `MultiTerm(GeometricTerm(drift, geometry), diffusion_term)`.

`CFEES25`, `CFEES27`, and `CFCosineEES` also work directly with SDE `MultiTerm`s. They are Stratonovich solvers, so the supplied drift and diffusion should describe the Stratonovich SDE in frame coordinates.

For the cosine EES family, use `CFCosineEES(reversible_order=11)` with any odd integer at least 3. The reversible order controls round-trip accuracy; forward accuracy remains second order. All members use a 2N low-storage recurrence and support `diffrax.ReversibleAdjoint()`. For example, `CFCosineEES(9)` uses five stages and `CFCosineEES(11)` uses six. This family has no embedded error estimate, so use fixed step sizes.

### Adaptive CF-EES steps

`CFEES25` and `CFEES27` provide an embedded 2(1) pair for ODEs. Enable adaptive stepping by passing a controller to `diffeqsolve`:

```python
sol = diffrax.diffeqsolve(
    term,
    CFEES25(),  # CFEES27() also supports the embedded pair.
    t0=0.0,
    t1=1.0,
    dt0=0.01,
    y0=jnp.eye(3),
    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
)
```

## Install

From a checkout, including the required pinned Diffrax fork:

```bash
pip install .
```

For development:

```bash
uv sync --extra dev
```

## Limitations

`RKMK` requires the selected chart to implement the inverse differential needed by the wrapped solver. On `SO(n)` and `Sphere(n)`, it uses the Cayley transform as an exact local coordinate map, together with its closed-form inverse differential, at every wrapped solver order. Sphere pullbacks operate through SO(n), avoiding the singular Jacobian of redundant sphere generators. This is distinct from using Cayley as an order-2 approximation to the exponential in a retraction method; see [Iserles and Zanna's Cayley-transform RKMK construction](https://doi.org/10.1112/S1461157000000206). Products delegate inverse differentials to their factors. Other geometries and charts use a generic dense-Jacobian fallback, which is accurate but can be expensive in high dimensions.

Intermediate saved samples and dense output preserve the manifold through chart interpolation, but do not inherit the solver's high-order accuracy between steps.

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

## Convergence examples

The deterministic and stochastic examples in `docs/examples` use 80-digit `decimal.Decimal` arithmetic by default, with crimson error markers, blue reference slopes, and PNG/PDF/JSON exports. Both plot CF-EES orders (2,5), (2,7), and (2,9), with the last corresponding to `CFCosineEES(9)`. Stochastic runs produce one figure for each Hurst parameter, with a single column ordered as 25, 25 recovery, 27, 27 recovery, 29, 29 recovery. Results go to `docs/examples/outputs`; `--precision`, `--min-power`, `--max-power`, and `--output-dir` configure either experiment.

```bash
python docs/examples/solver_convergence.py
python docs/examples/solver_convergence.py --include-cg
python docs/examples/stochastic_convergence.py
# Smaller stochastic run:
python docs/examples/stochastic_convergence.py --num-paths 4 --path-power 10 --max-power 8
```

These scripts evaluate the solver formulas and Georax's SO(3) charts independently in Decimal. Deterministic forward errors use an independent Decimal power-series reference. CF-EES backward errors are round-trip defects from the computed forward endpoint. With `--include-cg`, CG2 and CG4 instead integrate backward from the reference endpoint, matching the original experiment; these plots use a separate `_with_cg` filename.

CG4 coefficients are recomputed at the requested precision by solving all 13 fourth-order conditions, starting from the [published five-stage table](https://ancs.eng.buffalo.edu/pdf/ancs_papers/2012/geom_int.pdf). Its two free parameters, `a21 = 0.8177227988124852` and `a31 = 0.3199876375476427`, are fixed exactly; the remaining coefficients define a nearby high-precision tableau. The reproducible refinement is in `docs/examples/_decimal_cg4.py` and follows the [Crouch–Grossman order theory](https://doi.org/10.1023/A:1022325426017).

Stochastic runs use nested samples of the same two-channel Brownian or fractional Brownian path, with `--hurst 0.4 0.5 0.6` and `--seed 0` by default. Gaussian/FFT sampling uses binary64; samples are converted exactly to Decimal before integration, and Brownian scaling and accumulation also use Decimal. Forward errors compare against a finite finer-grid solution, and each curve averages log-errors across paths, as in the original example. JSON files include individual path errors, coefficients, precision, reference details, and measured slopes. Points below the Decimal roundoff threshold are shown in grey and excluded from fitting. The full default stochastic ensemble uses 12 paths with 8192 steps each and takes longer than the smaller run above.

## Benchmarking

```bash
uv run pytest tests/benchmarks -m benchmark -k 'test_runtime or test_grad_runtime' -s
```
