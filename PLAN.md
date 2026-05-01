# lfox plan

## Direction

`lfox` stays on JAX as the primary backend. The core abstraction — `Lattice` + `LatticeField` as immutable pytree-leaf modules, JIT-friendly `nn_field`, composable `Action` — is the right shape for a flexible LGT library and is worth pushing forward.

The design goal is composability and ease of modification (new theories, novel field content, exotic gauge groups) at modest cluster scale, where specialized C++ frameworks (Grid/QUDA/Chroma) don't compete because they don't cover the target physics. Reasonable performance on clusters via JAX sharding is the scaling story; leadership-class facility competitiveness is explicitly *not* a goal.

Apple Silicon is a desired desktop target. `jax-metal` is effectively dead and not worth relying on. The medium-term hedge is to **own the module abstraction** (drop direct `eqx.Module` inheritance in core types) so a second backend — MLX today, IREE/StableHLO via PJRT later — becomes a contained project rather than an architectural overhaul. No commitment yet to building the second backend; the durable investment is the seam.

ML-library interop is a plus, not a driver. Most LGT+ML prior art is PyTorch (normalizing flows, neural samplers), but the autodiff-for-HMC-forces win is what makes JAX the right primary choice; we're not switching to PyTorch for ecosystem reasons.

## Principles

- **Freeze new physics until the foundation converges.** No gauge fields, no new fermion machinery, until the legacy migration is done and tests exist.
- **Own the module abstraction.** Core types should not directly inherit `eqx.Module`. A thin frozen-dataclass + pytree-registration shim of our own gives the same ergonomics and keeps backends swappable.
- **Cluster sharding stays JAX-only and lives *above* any future backend seam.** Don't try to abstract `pmap`/`shard_map` across backends; MLX has no real distributed story and pretending otherwise is a trap.
- **Tests pin physics invariants, not implementations.** Translation invariance, leapfrog reversibility, `<exp(-ΔH)> = 1`, BC winding signs, solver convergence on known matrices. Cheap to write, enables aggressive refactors.

## Ordered work

### 1. Infrastructure hygiene (quick, unblocks everything)

- Consolidate package management on `uv`. `pyproject.toml` is the single source of truth; prune stale `requirements*.txt` / `setup.py` / `environment.yml` / conda artifacts from earlier experiments. Pin Python version in `[tool.uv]`. Document `uv sync` and `uv sync --extra metal` in the README.
- Adopt `jupytext` for any notebook that's load-bearing (Schaefer reproductions, anything that should be reviewable or diffable). Pair as `.ipynb` + `.py` (percent format); gitignore `.ipynb` outputs.
- Move genuinely scratch notebooks (`lfox_scratch*`, `refactor_scratch`, etc.) under `notebooks/scratch/`. Don't jupytext-ify throwaways.

### 2. Minimal pytest scaffolding + invariant tests

Populate `tests/` with a small `pytest` suite. Cheap unit tests pinning current behavior:

- `nn_field` boundary conditions (periodic and antiperiodic winding signs) on a small lattice.
- `Action.dS` vs finite-difference of `Action.S` on a free scalar action.
- Single short HMC trajectory with fixed seed — bit-reproducibility check.
- `Action` composition: `+`, `+=`, nested `sub_actions`, `params` merging.
- Leapfrog reversibility on a free field.
- `<exp(-ΔH)> = 1` over a small ensemble.
- `MRSolver` convergence on a known small matrix.

Heavy end-to-end physics tests (full Schaefer reproduction, multi-minute runs) stay opt-in: either as notebooks or `@pytest.mark.slow`.

### 3. Bug fixes (from `rewrite_thoughts.md`)

- `Action.__iadd__` returns `None` (`hmc.py:196-198`). Fix to `return self` or delete and rely on `+`.
- Off-by-one in `HMCEvolver.evolve` (`hmc.py:626,631,666`). `r_accept` indexing is wrong: index 0 unused, index `ntraj` OOB. Likely moot if `HMCEvolver` is deleted (see step 4).
- `Action.__copy__` has unreachable code after `return` (`hmc.py:211-226`).
- `LatticeField.__post_init__` uses `type(self.F) != jnp.array` (`lattice.py:149`) — always true. Use `isinstance(self.F, jax.Array)`.
- `MRSolver.solve` is a host-side Python `while` loop (`solver.py:19`). Convert to `jax.lax.while_loop` with a residual predicate so the inner loop doesn't pay dispatch cost per step.
- `HoneycombLattice.shift` FIXME — fix or mark explicitly unsupported.
- `Action.params` is `static=True` but mutated by `add_subaction` (`hmc.py:235`). Either freeze post-construction or stop marking static.

### 4. Finish the Old → new migration

- Migrate `Dirac4DFermionField` to inherit from the new `LatticeField` properly. Drop the manual `_tree_flatten` / `_tree_unflatten` registration in `spin.py:107-111` (those hooks only exist on `OldLatticeField`; the fermion code is currently silently riding the legacy path).
- Once nothing imports `OldLattice` / `OldLatticeField`, either delete them or quarantine in `legacy.py` so they don't pollute the JAX pytree registry on import.
- Delete `HMCEvolver`. Re-add its diagnostic instrumentation (`monitor`, `field_chain`, `obs_chain`) as a thin Python wrapper around `HMCRewrite`. Two parallel HMC classes is the wrong steady state.
- Resolve the list-vs-dict split at the `Action` boundary. Pick one — dicts everywhere with a lightweight rename layer for flavor aliasing is probably cleanest. Hybrid is the worst of both.

### 5. Own the module abstraction

Replace direct `eqx.Module` inheritance in `Lattice` / `LatticeField` / `Action` / integrators with a small in-tree `Module` base:

- Frozen dataclass-style.
- Explicit `tree_flatten` / `tree_unflatten` plus a `replace(**kw)` helper.
- Pytree registration goes through a backend-selectable hook (today: JAX; tomorrow: whatever).

This is the single architectural change that makes a second backend possible later. Doing it in the same pass as the legacy migration is much cheaper than coming back for a third pass — the tests from step 2 are the safety net that lets it happen aggressively.

`Equinox`-the-library can still be used for things it's uniquely good at (`filter_jit`-style helpers); the change is that *core types* no longer inherit from `eqx.Module`.

### 6. *Then* new physics

Gauge fields are the first real addition. `WilsonDiracOp.shift_fermion` is already the right hook for link variables. After that, the architecture has earned the right to grow.

## Deferred / optional

- **MLX backend.** Realistic to build on top of the owned `Module` abstraction; ~4–8 weeks of focused work plus ongoing ~10–20% maintenance tax. Not worth committing to until step 5 is done and Apple's signal on MLX investment is still positive.
- **IREE / StableHLO via PJRT as Apple Silicon path.** Watch-this-space. Currently bleeding edge; not turn-key for interactive Jupyter workflow. The owned-Module seam protects against either MLX or StableHLO not panning out — the seam is the durable bet, the specific backend is disposable.
- **PyTorch port.** Not recommended. Better Apple Silicon and ML interop, but the pure-functional `Action` / immutable `LatticeField` design swims upstream of PyTorch's mutable-module idiom. 2–3 month rewrite for a worse fit.
