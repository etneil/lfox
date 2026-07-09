# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`lfox` ("Lattice Fields Over jaX") is a JAX-based library for lattice field theory simulations: Markov-chain field evolution (HMC), lattice operators, and (in progress) fermion solvers / gauge fields. The core abstraction is an `eqx.Module`-backed `Lattice` + `LatticeField` pair that participates in JAX pytrees, supports `jax.jit` / `jax.grad`, and exposes overloaded arithmetic.

The repo is a research codebase: most exploration happens in the Jupyter notebooks at the repo root (e.g. `reproduce_schaefer*.ipynb`, `lfox_scratch*.ipynb`, `volume-ml.ipynb`). The installed package lives in `src/lfox/`.

## Environment & commands

- Python ≥ 3.12, managed with `uv` (lockfile at `uv.lock`).
- Sync env: `uv sync` (use `uv sync --extra metal` on Apple Silicon for `jax-metal`).
- Run a script in-env: `uv run python <script>`.
- Launch Jupyter: `uv run jupyter lab` (notebooks expect the package importable as `lfox`).
- Run tests: `uv run pytest`. `tests/` pins physics invariants (action normalization, autodiff force vs. analytic force, `<exp(-ΔH)> = 1`, RNG key threading, `Chain` bookkeeping). There is no CI, linter, or formatter configured.

## Architecture

### Lattice + LatticeField (`src/lfox/lattice.py`)

- `Lattice(eqx.Module)` holds `st_dims` (spacetime extents, a tuple) and `_dims` (physical dims; may differ from `st_dims` when there is a non-trivial unit cell, as in `HoneycombLattice`). `_bc_coords` is a precomputed `jnp.meshgrid` used for boundary-condition winding factors — that's why `__eq__`/`__hash__` are overridden to ignore it.
- Concrete lattices implement `shift(field, axis, shift)`. `SquareLattice` uses `jnp.roll`; `HoneycombLattice` adds A/B unit-cell handling on axis 0 (note the FIXME there — partially correct).
- `LatticeField(eqx.Module)` carries a `Lattice` (static field), the array `F`, boundary conditions `bc` (tuple of ±1 per axis, default periodic), and optional `indices` (extra trailing dims for spin/color/etc.). Arithmetic operators are overloaded via `_field_op` and dispatch on `getattr(other, 'F', other)`, so you can mix `LatticeField` with scalars or arrays freely.
- `nn_field(axis, shift=1)` is the primary kinematic primitive: nearest-neighbor shift with global boundary-condition winding factor applied. The hot path is `_nn_field`, a JIT-compiled staticmethod with the lattice/axis/shift/bc/st_dims passed as static args.
- `LatticeField` is immutable (Equinox); to produce a modified copy, use `copy_new_F(new_F)` (which round-trips through `dataclasses.replace`).

### Action (`src/lfox/action.py`)

- `Action(eqx.Module)` is the action functional. It lives at the package top level, next to `lattice.py`, because it is a physics abstraction rather than an evolution one — observables and (eventually) fermion force terms consume it too.
- Subclasses must implement a `@staticmethod _S(fields, params)` that takes a **list** of fields and a params dict and returns a per-site action density `LatticeField`. The instance methods `S_field`/`S`/`dS` accept **dicts** keyed by field name and remap to the list ordering via `self.field_names`. List-vs-dict at this boundary is intentional (enables aliasing fields for many-flavor setups); see `PLAN.md` for the plan to unify it.
- JIT decorator ordering matters: write `@staticmethod` *above* `@jax.jit` on `_S`, otherwise tracing breaks. This is called out in comments and is easy to get wrong.
- Actions compose with `+` / `+=` via `sub_actions`. `add_subaction` flattens nested sub-actions and merges `params` dicts (last-write-wins — there is a TODO to warn on collisions).

### Evolution (`src/lfox/evolution/`)

Import from the subpackage (`from lfox.evolution import HMC, Chain, LeapfrogIntegrator`), not from the individual modules.

- `base.py` — `Evolver(eqx.Module)` is the abstract Markov transition kernel: `evolve(fields, rng_key) -> (fields, monitor, rng_key)`. **Evolvers are pure.** They carry only what is constant across the chain (the action, algorithm hyperparameters); the fields and RNG key flow through as arguments. Never add mutable state to an `Evolver` — that is what made the old `HMCEvolver` unjittable. The base supplies a concrete `evolve_many` that runs the loop in a `jax.lax.scan`, which stacks whatever `monitor` dict the kernel returns, so the base never needs to know the diagnostic keys.
- `base.py` — `Chain` is the host-side driver: an ordinary stateful Python class (not an `eqx.Module`) owning the RNG key, current fields, saved configurations, diagnostics, and observable measurements. It steps in blocks equal to the GCD of `save_freq` and the observable frequencies, so it never steps over a trajectory at which something must be recorded. All compiled work happens inside `evolver.evolve_many`.
- `hmc.py` — `HMC(Evolver)`. `warmup` is a **static field**, not a call argument ("always accept" is a different transition kernel); build the variant with `as_warmup()`. Note that the accept/reject uniform must be drawn from an RNG key advanced past *every* momentum draw, or it is a deterministic function of the momenta it is testing; `tests/test_scalar_hmc.py` pins this.
- `integrators.py` — `LeapfrogIntegrator`, `OmelyanIntegrator` are `MDIntegrator` subclasses with `eps`, `Nstep` and a JIT-compiled `_integrate(delta_X, delta_P, X, P)`. `delta_X`/`delta_P` are closures supplied by the evolver — they are passed as static args (`static_argnums=(1,2)`), so re-creating them per call will trigger recompilation.
- Algorithm-specific requirements belong on the `Action`, not on `Evolver`. HMC needs `dS` (free, via autodiff, for any action); a heatbath would need local conditional distributions, a cluster update bond weights. Keep the `Evolver` interface thin.

### Fermions (`src/lfox/fermions/`)

- `spin.py`: DeGrand-Rossi gamma matrices (`Gamma_0..3,5`) and `Dirac4DFermionField` (a `LatticeField` with a fixed `indices=(4,)` Dirac index). Bilinears are JIT-compiled `jnp.einsum`s. **`import lfox.fermions.spin` currently raises**: it registers `_tree_flatten`/`_tree_unflatten`, which only ever existed on the now-deleted `OldLatticeField`. The whole fermion subpackage is stranded on the wrong side of the legacy migration; fixing it is `PLAN.md` step 4.
- `wilson.py`: `WilsonDiracOp` builds Dslash from gamma projectors and `nn_field`. `shift_fermion` is the gauge-coupling hook: override it to introduce link variables.
- `solver.py`: `MRSolver` (minimal-residual) for `D ψ = χ`. The `_MR_step` is JIT-compiled with `static_argnums=(0,)` — the solver instance is treated as static, so creating new solver instances per call is wasteful.

## Conventions / things to know

- Everything performance-critical is `jax.jit`-compiled; `static_argnums` is used liberally to keep `Lattice`/`Action`/integrator instances out of the traced args. Adding mutable fields to these `eqx.Module`s will break that.
- Boundary conditions: `bc` is a tuple of ±1 (only (anti-)periodic supported). Winding factor is `bc[axis]**winding` where `winding = (coords + shift) // L`.
- Field arithmetic and `nn_field` always return new `LatticeField` instances; never mutate `self.F` — `LatticeField` is a frozen `eqx.Module` and assignment raises.
- The pure/stateful boundary is the jit boundary. `Lattice`, `LatticeField`, `Action`, `MDIntegrator` and `Evolver` are pure `eqx.Module`s; `Chain` is the one stateful class, and it lives entirely outside jit.
- Notebooks at the repo root are scratch/research artifacts of varying staleness. `reproduce_schaefer*.ipynb` are reference physics reproductions; `refactor_scratch.ipynb` and `*_refactor.ipynb` track the in-progress legacy → Equinox migration. Treat them as references, not specifications.
