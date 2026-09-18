# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent-human collaboration

This file and the top level @README.md file should be treated as __human-only artifacts__; for agent partners, they are read-only and should not be modified.

## Project

`lfox` ("Lattice Fields Over jaX") is a JAX-based library for lattice field theory simulations: Markov-chain field evolution (HMC), lattice operators, and (in progress) fermion solvers / gauge fields. The core abstraction is an `eqx.Module`-backed `Lattice` + `LatticeField` pair that participates in JAX pytrees, supports `jax.jit` / `jax.grad`, and exposes overloaded arithmetic.

## Environment & commands

- Python ≥ 3.12, managed with `uv` (lockfile at `uv.lock`).
- Sync env: `uv sync` (use `uv sync --extra mps` on Apple Silicon for `jax-mps`, experimental).
- Run a script in-env: `uv run python <script>`.
- Run tests: `uv run pytest`. `tests/` pins physics invariants (action normalization, autodiff force vs. analytic force, `<exp(-ΔH)> = 1`, RNG key threading, `Chain` bookkeeping). There is no CI, linter, or formatter configured.

## Coding philosophy

Four core philosophies of developing `lfox`:

1. __Clean and concise.__  This is a library meant for use by a broader audience.  It is meant specifically for use in theoretical physics research.  As a result, it is essential that the code remains _clean_ and _concise_.  Anyone who is using the code should be able to read it and understand how it functions, so they can verify that the physics outputs will be what they expect.
2. __High-level and ergonomic.__  There are other lattice field theory codes written in lower-level compiled languages already on the market, but they mostly require a good amount of specialized knowledge to use properly.  One of the main reasons `lfox` is written in Python is to enable simple, high-level code.  When implementing user-facing functionality, ergonomics is the guiding principle; functions should be simple and natural to use.
3. __Flexible and extensible.__  Another motivation for working in Python is flexibility and extensibility.  `lfox` code should generally be written to be easily modified or specialized.  Our goal is _not_ to be the fastest code for doing lattice QCD at the physical point; instead, we want to be the _easiest_ code to get up and running with in a new theory or with a new calculation.
4. __Test-driven development.__ This is a code for computational physics research, so correctness is extremely important - we want to get science results that are reproducible and trustworthy out of this code.  When developing for `lfox`, follow test-driven development principles: new code must _always_ be accompanied with tests, ideally with the (initially failing) tests being written first.  Code-path tests are fine, but tests for physics/numerical correctness are better.  


## Testing

Three kinds of tests should be designed as part of the development of `lfox`.

1.  __Regression tests.__  These are automated tests which live in `tests/` and are invoked using `pytest`.  These are standard tests targeting behavior of individual parts of the code to make sure it works as expected and specified.  These should be kept __fast__, so that the regression test suite can be run easily as part of any code-development work.
2. __Integration tests.__  These are _physics_ tests which live in `integration/` and are designed to probe whether the simulation outputs produce known/expected results.  They also serve as demonstrations or examples of how to use `lfox`, appropriate for new users.  Instead of using `pytest`, these are designed as Jupyter notebooks, saved to the repository as Python scripts using `jupytext` (which also makes them accessible to agents if needed.)  These may be much more computationally demanding, but should be kept to a level that a new user on a workstation is able to run them in a reasonable amount of time.  These should _not_ be run automatically with all code changes, but only periodically and typically by a human user directly.
3. __Performance tests.__  These are tests of the efficiency and speed of the `lfox` implementation and live in `benchmarks/`.  Like integration tests these are on a slow duty-cycle, and should only be run periodically or when specifically trying to improve implementation details.  These may contain a mix of tests intended for different targets, in particular some at workstation-scale and some at cluster-scale targeting larger, more distributed workflows.  Obviously, tests should only be run on matching hardware, since not all tests will be appropriate for all machines.



## Architecture

These are old notes and may be supplanted as progress is made, and likely moved to a new file.  For now they are kept here for reference.

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
