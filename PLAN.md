# lfox plan

## Direction

`lfox` stays on JAX as the primary backend. The core abstraction — `Lattice` + `LatticeField` as immutable pytree-leaf modules, JIT-friendly `nn_field`, composable `Action` — is the right shape for a flexible LGT library and is worth pushing forward.

The design goal is composability and ease of modification (new theories, novel field content, exotic gauge groups) at modest cluster scale, where specialized C++ frameworks (Grid/QUDA/Chroma) don't compete because they don't cover the target physics. Reasonable performance on clusters via JAX sharding is the scaling story; leadership-class facility competitiveness is explicitly *not* a goal.

Apple Silicon is a desired desktop target. `jax-metal` is effectively dead and not worth relying on. The path forward is to **wait for upstream JAX-compatible backends** to mature — community MLX-as-JAX-backend efforts, IREE/StableHLO via PJRT — rather than build a parallel MLX backend ourselves. Those paths are JAX-compatible by construction, so no architectural seam is needed; staying idiomatic in JAX *is* the hedge. A narrower DLPack interop with MLX for post-sampling observables (outside autodiff, outside JIT) is a viable side path if a specific analysis kernel motivates it.

ML-library interop is a plus, not a driver. Most LGT+ML prior art is PyTorch (normalizing flows, neural samplers), but the autodiff-for-HMC-forces win is what makes JAX the right primary choice; we're not switching to PyTorch for ecosystem reasons.

## Principles

- **Freeze new physics until the foundation converges.** No gauge fields, no new fermion machinery, until the legacy migration is done and tests exist.
- **Stay idiomatic in JAX.** Core types continue to inherit `eqx.Module`. With a parallel-backend hedge no longer the strategy, Equinox earns its place — surface use here (`Module`, `field(static=...)`, `field(converter=...)`, `field(init=False)`) is small and the dependency is near-free. Revisit only if the upstream Apple-Silicon path collapses and we are forced to build a parallel backend.
- **Cluster sharding stays JAX-native.** `pmap`/`shard_map` are the scaling story. Nothing to abstract across.
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
- `Lattice.__eq__` raises `NotImplementedError` (`lattice.py:80-81`) instead of returning the `NotImplemented` singleton — `lattice == 5` explodes rather than returning `False`.
- `MRSolver.solve` is a host-side Python `while` loop (`solver.py:19`). Convert to `jax.lax.while_loop` with a residual predicate so the inner loop doesn't pay dispatch cost per step.
- `HoneycombLattice.shift` FIXME — fix or mark explicitly unsupported.
- `Action.params` is `static=True` but mutated by `add_subaction` (`hmc.py:235`). Either freeze post-construction or stop marking static.
- `LatticeField.copy_new_F` (`lattice.py:288-294`) — simplify the `cls.__new__` + `__dict__.update` + `dataclasses.replace` dance to a single `dataclasses.replace(self, F=new_F)`.
- Delete the commented-out alternate `nn_field` block (`lattice.py:254-279`).

### 4. Finish the Old → new migration

- Migrate `Dirac4DFermionField` to inherit from the new `LatticeField` properly. Drop the manual `_tree_flatten` / `_tree_unflatten` registration in `spin.py:107-111` (those hooks only exist on `OldLatticeField`; the fermion code is currently silently riding the legacy path).
- Once nothing imports `OldLattice` / `OldLatticeField`, either delete them or quarantine in `legacy.py` so they don't pollute the JAX pytree registry on import.
- Delete `HMCEvolver`. Re-add its diagnostic instrumentation (`monitor`, `field_chain`, `obs_chain`) as a thin Python wrapper around `HMCRewrite`. Two parallel HMC classes is the wrong steady state.
- Resolve the list-vs-dict split at the `Action` boundary. Pick one — dicts everywhere with a lightweight rename layer for flavor aliasing is probably cleanest. Hybrid is the worst of both.

### 5. *Then* new physics

Gauge fields are the first real addition. `WilsonDiracOp.shift_fermion` is already the right hook for link variables. After that, the architecture has earned the right to grow.

## Deferred / optional

- **Parallel MLX backend.** Considered and rejected: a dual backend either duplicates the core Module / HMC / integrator layer or forces a lowest-common-denominator `xnp` shim that loses JAX's jit / static_argnums / Equinox ergonomics. The kernels likeliest to benefit from hand-tuned MLX (Wilson Dslash, fermion solvers) sit inside HMC's gradient path, where calling MLX from JIT'd JAX requires a manual `custom_vjp` and risks host round-trip overhead via `pure_callback`. Revisit only if every upstream Apple-Silicon path fails to mature.
- **MLX-via-DLPack for post-sampling analysis.** Narrower and viable: bridge JAX↔MLX with DLPack for non-differentiated observables computed outside any JIT. Treat MLX as an analysis-side library that happens to share memory — no architectural changes required. Pursue only if a concrete observable kernel motivates it. Validate first that DLPack handoff is actually zero-copy on the same device; that's the load-bearing assumption.
- **Community MLX-as-JAX-backend / IREE / StableHLO via PJRT.** The actual hoped-for Apple-Silicon path. Watch-this-space; bleeding edge today, not turn-key for interactive Jupyter. By construction these are JAX-compatible, so no architectural seam is needed.
- **Own the module abstraction (drop `eqx.Module` from core types).** Was a hedge for parallel-backend swappability; with that off the table, the work is unmotivated. Reopen only if the upstream-backend bet fails *and* a parallel backend becomes necessary.
- **PyTorch port.** Not recommended. Better Apple Silicon and ML interop, but the pure-functional `Action` / immutable `LatticeField` design swims upstream of PyTorch's mutable-module idiom. 2–3 month rewrite for a worse fit.
