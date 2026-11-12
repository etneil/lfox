## Architecture

These are running architecture notes and may be supplanted as progress is made.  For now they are kept here for reference.  Last revised 2026-11-11, on branch `scalar`.

Package layout: `lattice.py` and `action.py` are the core; `evolution/` holds Markov-chain algorithms and the `Chain` driver; `stats.py` holds Monte Carlo statistics; `scalar/` holds theory-specific observables; `fermions/` is stranded (see below).

### Lattice + LatticeField (`src/lfox/lattice.py`)

- `Lattice(eqx.Module)` holds `st_dims` (spacetime extents, a tuple) and `_dims` (physical dims; may differ from `st_dims` when there is a non-trivial unit cell, as in `HoneycombLattice`).  It carries no JAX arrays, so it can be a static field and compares/hashes by value: the boundary-condition coordinate grid is computed on demand by `bc_coord(axis)`.  `volume` is the number of sites, including unit-cell structure.
- Concrete lattices implement `shift(field, axis, shift)`. `SquareLattice` uses `jnp.roll` (its red-black `rb_split`/`rb_combine` are stubs); `HoneycombLattice` adds A/B unit-cell handling on axis 0 (note the FIXME there — partially correct).
- `LatticeField(eqx.Module)` carries a `Lattice` (static field), the array `F`, boundary conditions `bc` (static tuple of ±1 per axis, default periodic), and optional `indices` (static; extra trailing dims for spin/color/etc.).  A scalar `F` is broadcast to the lattice shape, so `LatticeField(lattice=L, F=1)` is a unit field.  Arithmetic operators are overloaded via `_field_op` and dispatch on `getattr(other, 'F', other)`, so you can mix `LatticeField` with scalars or arrays freely.
- `nn_field(axis, shift=1)` is the primary kinematic primitive: nearest-neighbor shift with global boundary-condition winding factor applied. The hot path is `_nn_field`, a JIT-compiled staticmethod with the lattice/axis/shift/bc/st_dims passed as static args.
- `d()` is the number of spacetime dimensions; `dims()` includes any internal `indices`.
- `LatticeField` is immutable (Equinox); to produce a modified copy, use `copy_new_F(new_F)` (which round-trips through `dataclasses.replace`).

### Action (`src/lfox/action.py`)

The module docstring and the docstring of `tests/test_action.py` are the authoritative design statement; this is a summary.

- Two classes, one job each.  **`Term` is what you write**: subclass it and supply `density` (per-site action density, a `LatticeField`) or, for a term too nonlocal to have one, `total` (a scalar).  **`Action` is what you evaluate**: a flat, final sum of terms, built by `Action(*terms)`, `+`, or `sum(...)`.  Never subclass `Action`; a preset theory is a function returning one.  A `Term` has no `S`/`dS`.
- **Fields are named by the hook signature.**  `def density(self, phi)` makes `phi` a *role* of the term, and by default also its external field name.  `rebind` renames: `Term.rebind` is keyed by role, `Action.rebind` by external name (roles are only unique within one term class).  `replicate(n, *names)` is the many-flavor loop (suffixes `_0 .. _{n-1}`).
- **Couplings are ordinary dataclass fields**, hence pytree leaves: retuning one (`with_param`) does not retrace, and `jax.grad` can differentiate through it.  Anything that steers Python control flow inside a hook, or is not numeric, must be `eqx.field(static=True)` (checked at construction).  `lambda` is a Python keyword, so the φ⁴ coupling is spelled `lamb` by convention.
- Public API on `Action`: `S_field(fields)`, `S(fields)` (also `action(fields)`), `dS(fields, wrt=None)` and `sample(fields, key)`, all taking/returning **dicts keyed by external field name**.  `dS` is the gradient dS/dφ, not the force, by autodiff unless a term supplies a hand-written `gradient` hook (such a term is never also differentiated).
- **Sampled vs evolved fields.**  A term may declare `sampled = ('eta',)` and a `draw(key, ...)` hook giving an exact conditional heatbath (pseudofermions).  `Action.sampled_fields` / `evolved_fields` follow; `dS` defaults to the evolved fields.  `sample` gives each drawing term its own key.
- Composition guards: two terms with the same `name` (label, default class name) on the same fields raise `DuplicateTermError`; distinct `label`s declare a deliberate duplicate.  `S[label]` looks up a term.
- Consumers accept a lone term via `eqx.field(converter=Action)`, so downstream code only ever sees an `Action`.

### Evolution (`src/lfox/evolution/`)

Import from the subpackage (`from lfox.evolution import HMC, Chain, LeapfrogIntegrator`), not from the individual modules.

- `base.py` — `Evolver(eqx.Module)` is the abstract Markov transition kernel: `evolve(fields, rng_key) -> (fields, monitor, rng_key)`. **Evolvers are pure.** They carry only what is constant across the chain (the action, algorithm hyperparameters); the fields and RNG key flow through as arguments. Never add mutable state to an `Evolver` — that is what made the old `HMCEvolver` unjittable. The base supplies a concrete `evolve_many` that runs the loop in a `jax.lax.scan`, which stacks whatever `monitor` dict the kernel returns, so the base never needs to know the diagnostic keys.  `as_warmup()` returns the thermalization variant (identity by default).
- `base.py` — `Chain` is the host-side driver: an ordinary stateful Python class (not an `eqx.Module`) owning the RNG key, current fields, saved configurations, diagnostics, and measurements.
  - `measure=[f, (g, freq), ...]`: each measurement function is `f(fields, action) -> {name: value}`, run every trajectory or every `freq`.  `Chain` shape-checks (`jax.eval_shape`) and jit-compiles them at construction, so a function must be traceable (no `float()` on a traced value) and a name returned by two functions is an error up front.
  - Steps in blocks of the GCD of `save_freq` and the measurement frequencies, so it never steps over a trajectory at which something must be recorded; `run(ntraj)` requires `ntraj` to be a multiple of that block.
  - Reading results: `column(name)` (float64 array, leading axis = MC time), `column_traj(name)`, and `columns(*names)` which aligns several on their shared trajectories (what a multi-column jackknife needs).  `acceptance()` and `monitor` hold diagnostics.
  - `warmup(n)` uses `evolver.as_warmup()`, records diagnostics only, and advances `warmup_traj`; `traj` counts production trajectories only.
- `hmc.py` — `HMC(Evolver)`. `warmup` is a **static field**, not a call argument ("always accept" is a different transition kernel); build the variant with `as_warmup()`.  Each trajectory first heatbaths the action's sampled fields (if any), then refreshes momenta for the evolved fields only, then integrates and accepts/rejects.  The accept/reject uniform must be drawn from an RNG key advanced past *every* momentum draw, or it is a deterministic function of the momenta it is testing; `tests/test_scalar_hmc.py` pins this, and `integration/pseudofermion_marginal.py` checks the heatbath/MD composition against an exact marginal.
- `integrators.py` — `LeapfrogIntegrator`, `OmelyanIntegrator` are `MDIntegrator` subclasses with `eps`, `Nstep` and a JIT-compiled `_integrate(delta_X, delta_P, X, P)` (behind a thin `integrate` wrapper kept for profiling).  `X`/`P` are dicts of fields.  `delta_X`/`delta_P` are closures supplied by the evolver — they are passed as static args (`static_argnums=(1,2)`), so re-creating them per call outside an enclosing jit will trigger recompilation.
- Algorithm-specific requirements belong on the `Action`, not on `Evolver`. HMC needs `dS` (free, via autodiff, for any action); exact heatbaths come from `Term.draw`; a cluster update would need bond weights. Keep the `Evolver` interface thin.

### Statistics (`src/lfox/stats.py`)

Host-side NumPy at float64 on *columns* (leading axis = MC time, as from `Chain.column`).

- `jackknife(estimator, *columns, bin_size=1, **constants) -> (estimate, error)`.  `estimator` is the ordinary function of raw samples; positional columns are resampled together (must be aligned); keyword `constants` pass through.  Deletes blocks of *raw* samples rather than binning first, which keeps nonlinear moments right.  Cost is N / `bin_size` + 1 evaluations; `bin_size` is the lever against τ_int, and `bin_size=1` is right for data already binned (bin the primary observables, not the field a nonlinear estimator consumes).
- `tau_int(series) -> TauInt` with Sokal's automatic window and Madras–Sokal error.  `TauInt(tau, error, window, resolved, gamma)` prints as `TauInt(tau=4.52(18), window=23)`; an unresolved window prints `tau>=` and is only a lower bound.

### Scalar observables (`src/lfox/scalar/`)

- Two kinds, deliberately separate.  **Per-configuration** observables take a `LatticeField` and are jittable — what a `Chain` measurement function returns: `magnetization` (m = M/V), `nn_product`, `structure_factor(phi, n)` (G(p), not connected), `correlator(phi, axis)` (zero-momentum C(t), not connected).  **Ensemble** estimators take columns and are nonlinear in averages, so their errors come from `jackknife`: `susceptibility(m, volume)`, `binder`, `xi_2nd(m, Gp, volume, L)`, `effective_mass(C)`.
- Broken-phase χ is `susceptibility(np.abs(m), volume)`.
- Ensemble estimators return a quiet `nan` where the quantity does not exist (`xi_2nd` outside χ ≥ ⟨G(p_min)⟩ > 0; `effective_mass` at t = 0 and where noise pushes the cosh ratio below 1), rather than `inf` or a warning per jackknife deletion.
- O(N) models are a near-term target; whether they extend this module (N = 1 as the ordinary scalar) or get their own subpackage is undecided.

### Fermions (`src/lfox/fermions/`)

- `spin.py`: DeGrand-Rossi gamma matrices (`Gamma_0..3,5`) and `Dirac4DFermionField` (a `LatticeField` with a fixed `indices=(4,)` Dirac index). Bilinears are JIT-compiled `jnp.einsum`s. **`import lfox.fermions.spin` currently raises** (`AttributeError: ... '_tree_flatten'`): it registers pytree hooks that only ever existed on the now-deleted `OldLatticeField`. `import lfox.fermions` itself is clean because nothing is re-exported. The whole fermion subpackage is stranded on the wrong side of the legacy migration; fixing it is `PLAN.md` step 4.
- `wilson.py`: `WilsonDiracOp` builds Dslash from gamma projectors and `nn_field`. `shift_fermion` is the gauge-coupling hook: override it to introduce link variables.
- `solver.py`: `MRSolver` (minimal-residual) for `D ψ = χ`. The `_MR_step` is JIT-compiled with `static_argnums=(0,)` — the solver instance is treated as static, so creating new solver instances per call is wasteful.
- The pseudofermion *mechanics* (sampled fields, `draw`, hand-written `gradient`) already exist in `Action` and are exercised by `integration/pseudofermion_marginal.py`; what is missing is the Dirac operator side.

### Tests and integration

- `tests/phi4.py` — `ScalarTerm`, the reference φ⁴ term, with an analytic gradient for checking autodiff.  `tests/conftest.py` provides fixtures (`scalar_action`, `lat4`, `phi4`).
- `tests/free_field.py` — exact, independent samples of the free (λ = 0) field via FFT, plus closed-form χ, ξ₂, C(t) and pole mass: the reference values the observables are tested against.
- `integration/reproduce_schaefer.py` — reproduction of Schaefer's φ⁴ exercises; `integration/pseudofermion_marginal.py` — heatbath + HMC composition against an exact marginal.  Both are jupytext-paired notebooks.

## Conventions / things to know

- Everything performance-critical is `jax.jit`-compiled.  Modules (`Action`, `Evolver`, integrators) are passed to jitted methods as pytrees: numeric fields (couplings, `eps`) are traced leaves, and `eqx.field(static=True)` fields go into the compilation key.  Changing a static field retraces; changing a coupling does not.  Adding mutable state to these `eqx.Module`s will break this.
- `static_argnums` is still used for genuinely static arguments: lattice/axis/shift in `_nn_field`, the closures passed to `_integrate`, the field tuple in `dS`, `traj` in `evolve_many`.
- Boundary conditions: `bc` is a tuple of ±1 (only (anti-)periodic supported). Winding factor is `bc[axis]**winding` where `winding = (coords - shift) // L` (signs show up at the target site).
- Field arithmetic and `nn_field` always return new `LatticeField` instances; never mutate `self.F` — `LatticeField` is a frozen `eqx.Module` and assignment raises.
- Field collections at every public boundary are dicts keyed by external field name; only author hooks see roles.
- The pure/stateful boundary is the jit boundary. `Lattice`, `LatticeField`, `Term`, `Action`, `MDIntegrator` and `Evolver` are pure `eqx.Module`s; `Chain` is the one stateful class, and it lives entirely outside jit.  Statistics (`lfox.stats`, ensemble estimators) are host-side NumPy, after the chain.
- Enable `jax_enable_x64` for physics runs; the tests and examples do.
