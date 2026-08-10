# lfox

[![CI](https://github.com/etneil/lfox/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/etneil/lfox/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lfox.svg)](https://pypi.org/project/lfox/)
[![Python versions](https://img.shields.io/pypi/pyversions/lfox.svg)](https://pypi.org/project/lfox/)
[![License](https://img.shields.io/pypi/l/lfox.svg)](https://github.com/etneil/lfox/blob/main/LICENSE.txt)

**L**attice **F**ields **O**ver ja**X** — a JAX-based library for lattice field theory
simulations: Markov-chain field evolution (HMC), lattice operators, and observable
measurement.

_Why?_ JAX provides two very useful things for lattice simulations:

1. __Autodiff__.  Implement the action as a Python function, JAX computes the
   necessary gradients for HMC or other algorithms automatically.
2. __Flexible backend__.  JAX can target GPUs and its backend for sharding can
   handle communications and scale up beyond a single machine.

Since JAX is focused on supporting machine learning applications, `lfox` is also
naturally interoperable with machine-learning workflows implemented in JAX.

The code being fully Python-based also means that it should be easy to work with and to
modify; this will probably never be the most efficient lattice code out there, but
the goal is "good enough for prototyping" or working with toy or exotic theory targets.

The core abstraction is an [Equinox](https://github.com/patrick-kidger/equinox)-backed
`Lattice` + `LatticeField` pair that participates in JAX pytrees, so everything composes
with `jax.jit` and `jax.grad`.

> **Status: early alpha.** This is a research codebase under active development. The API is not
> stable, and `lfox.fermions` is incomplete (see [Roadmap](#roadmap)). Pin an exact
> version if you depend on it.

## Installation

```console
pip install lfox
```

or, with [uv](https://docs.astral.sh/uv/):

```console
uv add lfox
```

On Apple Silicon you can optionally pull in the experimental Metal backend:

```console
pip install "lfox[mps]"
```

JAX is installed as a plain CPU build by default. For CUDA or TPU, install the
appropriate [JAX wheel](https://docs.jax.dev/en/latest/installation.html) for your
platform alongside `lfox`.

## Quickstart

A phi^4 scalar field on a 4³ lattice, evolved with hybrid Monte Carlo:

```python
import jax
import jax.numpy as jnp

import lfox.lattice as lat
from lfox.action import Action
from lfox.evolution import HMC, Chain, LeapfrogIntegrator

jax.config.update("jax_enable_x64", True)


class ScalarAction(Action):
    # NOTE: @staticmethod must sit *above* @jax.jit, or tracing breaks.
    @staticmethod
    @jax.jit
    def _S(fields, params):
        phi = fields[0]
        S = phi**2
        for ax in range(phi.d()):
            S -= 2 * params["kappa"] * phi * phi.nn_field(ax)
        S += params["lambda"] * (phi**2 - 1) ** 2
        return S


lattice = lat.SquareLattice(st_dims=(4, 4, 4))
phi = lat.LatticeField(lattice=lattice, F=jnp.ones((4, 4, 4)))

action = ScalarAction(field_names=["phi"], params={"kappa": 0.18169, "lambda": 1.3282})
hmc = HMC(action=action, integrator=LeapfrogIntegrator(eps=0.1, Nstep=10))

chain = Chain(
    evolver=hmc,
    seed=1,
    init_fields={"phi": phi},
    observables={"magn": lambda fields, params: float(jnp.mean(fields["phi"].F))},
)
chain.warmup(100)
chain.run(1000)

print(f"acceptance: {chain.acceptance():.3f}")
print(f"<phi>: {sum(chain.obs_chain['magn']) / len(chain.obs_chain['magn']):.5f}")
```

Note that only `_S` was written by hand. The force `dS` used by the integrator is
obtained by differentiating it.

## What's in the box

- **`lfox.lattice`** — `Lattice` and `LatticeField`, with overloaded arithmetic 
  and `nn_field(axis, shift)` for nearest-neighbor hops with (anti-)periodic boundary conditions.
- **`lfox.action`** — `Action`, the action functional. Subclass it and implement `_S`;
  you get `S`, `S_field`, and autodiff `dS` for free. Actions compose with `+`.
- **`lfox.evolution`** — `HMC` (a pure, jittable Markov transition kernel),
  `LeapfrogIntegrator` / `OmelyanIntegrator`, and `Chain`, the host-side driver that owns
  the RNG key, saved configurations, diagnostics, and observable measurement.
- **`lfox.fermions`** — Dirac spin structure, the Wilson operator, and a minimal-residual
  solver. **Experimental and currently non-functional** — this subpackage has not yet been
  migrated to the current `LatticeField`, and importing it raises.

## Roadmap

- Fermions!
- Gauge fields!
- Group/representation machinery
- Other update algorithms
- Proper testing of scaling and GPU backend
- A richer measurement / analysis layer

## Development

```console
git clone https://github.com/etneil/lfox
cd lfox
uv sync          # installs the package plus test and notebook dependencies
uv run pytest
```

The test suite pins physics invariants rather than implementation details: action
normalization, autodiff force against the analytic force, `<exp(-ΔH)> = 1`, RNG key
threading, and `Chain` bookkeeping.

Exploratory work lives in Jupyter notebooks at the repo root, paired to `py:percent`
scripts via [jupytext](https://jupytext.readthedocs.io/). Launch with `uv run jupyter lab`.

## License

`lfox` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html)
license.
