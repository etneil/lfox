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
- No build/lint/test tooling is configured. `tests/` exists but is empty (only `__init__.py`); there is no `pytest` config, CI, or formatter. Don't invent a test command — say so if asked to run tests.

## Architecture

### Lattice + LatticeField (`src/lfox/lattice.py`)

- `Lattice(eqx.Module)` holds `st_dims` (spacetime extents, a tuple) and `_dims` (physical dims; may differ from `st_dims` when there is a non-trivial unit cell, as in `HoneycombLattice`). `_bc_coords` is a precomputed `jnp.meshgrid` used for boundary-condition winding factors — that's why `__eq__`/`__hash__` are overridden to ignore it.
- Concrete lattices implement `shift(field, axis, shift)`. `SquareLattice` uses `jnp.roll`; `HoneycombLattice` adds A/B unit-cell handling on axis 0 (note the FIXME there — partially correct).
- `LatticeField(eqx.Module)` carries a `Lattice` (static field), the array `F`, boundary conditions `bc` (tuple of ±1 per axis, default periodic), and optional `indices` (extra trailing dims for spin/color/etc.). Arithmetic operators are overloaded via `_field_op` and dispatch on `getattr(other, 'F', other)`, so you can mix `LatticeField` with scalars or arrays freely.
- `nn_field(axis, shift=1)` is the primary kinematic primitive: nearest-neighbor shift with global boundary-condition winding factor applied. The hot path is `_nn_field`, a JIT-compiled staticmethod with the lattice/axis/shift/bc/st_dims passed as static args.
- `LatticeField` is immutable (Equinox); to produce a modified copy, use `copy_new_F(new_F)` (which round-trips through `dataclasses.replace`).
- **Legacy classes**: `OldLattice` and `OldLatticeField` are still in the same file and registered as pytree nodes. New code uses the Equinox versions; do not extend the `Old*` classes unless explicitly fixing legacy behavior.

### Action + HMC (`src/lfox/evolution/hmc.py`)

- `Action(eqx.Module)` is the action functional. Subclasses must implement a `@staticmethod _S(fields, params)` that takes a **list** of fields and a params dict and returns a per-site action density `LatticeField`. The instance methods `S_field`/`S`/`dS` accept **dicts** keyed by field name and remap to the list ordering via `self.field_names`. List-vs-dict at this boundary is intentional (enables aliasing fields for many-flavor setups).
- JIT decorator ordering matters: write `@staticmethod` *above* `@jax.jit` on `_S`, otherwise tracing breaks. This is called out in comments and is easy to get wrong.
- Actions compose with `+` / `+=` via `sub_actions`. `add_subaction` flattens nested sub-actions and merges `params` dicts (last-write-wins — there is a TODO to warn on collisions).
- Two HMC implementations live side by side:
  - `HMCEvolver(Evolver)`: original stateful evolver. Mutates `self.fields`, `self.field_chain`, `self.monitor` in a Python loop. Easier to instrument; harder to JIT end-to-end.
  - `HMCRewrite(eqx.Module)`: pure-functional rewrite. `evolve` and `evolve_many` are JIT-compiled (with `static_argnums` for `warmup`/`return_pi`/`traj`); `evolve_many` runs the trajectory loop inside `jax.lax.fori_loop`. Prefer this for new performance-sensitive code.
- Integrators (`LeapfrogIntegrator`, `OmelyanIntegrator`) are `MDIntegrator` subclasses with `eps`, `Nstep` and a JIT-compiled `_integrate(delta_X, delta_P, X, P)`. `delta_X`/`delta_P` are closures supplied by the HMC class — they are passed as static args (`static_argnums=(1,2)`), so re-creating them per call will trigger recompilation.

### Fermions (`src/lfox/fermions/`)

- `spin.py`: DeGrand-Rossi gamma matrices (`Gamma_0..3,5`) and `Dirac4DFermionField` (a `LatticeField` with a fixed `indices=(4,)` Dirac index). Bilinears are JIT-compiled `jnp.einsum`s. Note this file still uses the legacy `_tree_flatten`/`_tree_unflatten` registration — it's only correct against `OldLatticeField`, so the fermion code currently lives on the legacy path.
- `wilson.py`: `WilsonDiracOp` builds Dslash from gamma projectors and `nn_field`. `shift_fermion` is the gauge-coupling hook: override it to introduce link variables.
- `solver.py`: `MRSolver` (minimal-residual) for `D ψ = χ`. The `_MR_step` is JIT-compiled with `static_argnums=(0,)` — the solver instance is treated as static, so creating new solver instances per call is wasteful.

## Conventions / things to know

- Everything performance-critical is `jax.jit`-compiled; `static_argnums` is used liberally to keep `Lattice`/`Action`/integrator instances out of the traced args. Adding mutable fields to these `eqx.Module`s will break that.
- Boundary conditions: `bc` is a tuple of ±1 (only (anti-)periodic supported). Winding factor is `bc[axis]**winding` where `winding = (coords + shift) // L`.
- Field arithmetic and `nn_field` always return new `LatticeField` instances; never mutate `self.F` in new code (the `Old*` legacy classes do, the Equinox versions do not).
- Notebooks at the repo root are scratch/research artifacts of varying staleness. `reproduce_schaefer*.ipynb` are reference physics reproductions; `refactor_scratch.ipynb` and `*_refactor.ipynb` track the in-progress legacy → Equinox migration. Treat them as references, not specifications.
