# lfox rewrite thoughts

Bottom line up front: the *core* abstraction (Equinox-backed `Lattice` + `LatticeField` participating in pytrees, with JIT-friendly `nn_field` and a composable `Action`) is the right shape for a JAX lattice library — it's worth pushing forward. But there's enough half-finished migration in the tree that adding new physics on top will keep getting harder. Freeze features and converge before doing more.

## Real bugs (not just style)

- **`Action.__iadd__` returns `None`** (`hmc.py:196-198`). After `a += b`, `a` is `None`. Either fix to `return self` or delete and rely on `+`.
- **Off-by-one in `HMCEvolver.evolve`** (`hmc.py:626,631,666`). `r_accept` is shape `(ntraj,)`, but the loop uses `r_accept[traj]` with `traj ∈ [1, ntraj]` — index `0` is unused, index `ntraj` is OOB.
- **`HMCEvolver` mutates `LatticeField.F`** (`hmc.py:669`: `self.fields[fname].F = prev_fields[fname]`). `LatticeField` is an `eqx.Module` (frozen). This path is dead-on-arrival against the new field type — which is presumably why `HMCRewrite` exists, but `HMCEvolver` is still in the file as if usable.
- **Fermion code rides the legacy pytree path.** `Dirac4DFermionField` inherits from the new `LatticeField` but is registered with `_tree_flatten`/`_tree_unflatten` (`spin.py:107-111`), which only exist on `OldLatticeField`. Either it's broken or it's silently working by accident.
- **`Action.__copy__`** has unreachable code after the `return` (`hmc.py:211-226`).
- **`HoneycombLattice.shift`** has a flagged FIXME — the subclass isn't actually usable.

## Architectural calls — what to keep

- **`eqx.Module` for `Lattice`/`LatticeField`/`Action`/integrators**, with the `Lattice` carried as `static=True` on fields. Right call. The custom `__eq__`/`__hash__` on `Lattice` (ignoring `_bc_coords`) is the correct workaround for caching JAX arrays on a static field.
- **`_nn_field` as a JIT'd staticmethod with everything it needs as static args.** Slightly awkward signature, but it's how you avoid retracing per call.
- **Composable `Action` via `sub_actions` and `+`.** Good shape for multi-term actions (gauge + fermion + pseudofermion).
- **Pure-functional `HMCRewrite` with `lax.fori_loop` in `evolve_many`.** This is the right end state.

## Architectural calls — what to reconsider

- **List-vs-dict split at the `Action` boundary.** `_S` takes lists, `S_field`/`S` take dicts, and `S_field` does the remap via `field_names`. The justification (flavor aliasing) is real, but in practice every callsite has to keep ordering invariants in their head, and `add_subaction` merging `params` with last-write-wins (there's a TODO there) is going to bite. Two cleaner options: (a) commit to dicts everywhere and handle aliasing with a lightweight rename layer, or (b) commit to lists everywhere and keep names as a parallel tuple. Hybrid is the worst.
- **Two HMC classes coexisting.** Pick `HMCRewrite`, delete `HMCEvolver`, and re-add the diagnostic instrumentation (`monitor`, `field_chain`, `obs_chain`) as a thin Python wrapper around it. Right now both exist and only one works against the new types.
- **`OldLattice`/`OldLatticeField` in the same file as the new ones.** They're still pytree-registered, which means imports pollute the JAX pytree registry whether you use them or not. Move them to a `legacy.py` you can stop importing, or just delete once fermions are migrated.
- **`MRSolver.solve` is a Python `while` loop** (`solver.py:19`). The *step* is JIT'd, but the loop is host-side, so each step pays a dispatch cost and convergence is checked on the host. For a fermion solver in an HMC inner loop this is the wrong tradeoff — `jax.lax.while_loop` with a residual predicate is the standard pattern.
- **No tests.** This is the biggest leverage point for a physics codebase. Don't need a lot — a half-dozen tests pinning invariants (translation invariance of the action, leapfrog reversibility on a free field, HMC `<exp(-ΔH)> = 1`, MR solver convergence on a known matrix, BC winding sign on a small lattice) would let aggressive refactors proceed without breaking physics. Right now the only safety net is "does the notebook still reproduce Schaefer."
- **Notebook-driven development at this scale is hurting.** Several of the root notebooks are >300KB; the legacy/rewrite duality in `src/` looks like the natural consequence of "edit notebook, copy stable bits into `src/`, never finish the cutover." Worth pulling the stable kernels into `src/lfox/` modules and keeping notebooks thin.

## Smaller cleanups while you're in there

- `LatticeField.__post_init__` uses `type(self.F) != jnp.array` (`lattice.py:149`). `jnp.array` is a function, not a type — this comparison is always `True`. Use `isinstance(self.F, jax.Array)` or just always broadcast.
- `LeapfrogIntegrator._integrate` / `OmelyanIntegrator._integrate` pass `delta_X`/`delta_P` as static args (`static_argnums=(1,2)`) — this means *each* call to `HMCEvolver.delta_mom()` (which builds a fresh closure) triggers a recompile. `make_deltas()` saves them, which is the right intent, but `HMCRewrite.delta_mom`/`delta_fields` rebuild closures every `evolve` — worth checking with `jax.jit`'s cache.
- `Action.params` being `static=True` while it's mutated in `add_subaction` (`hmc.py:235`) is a footgun. Either freeze `params` or stop mutating it post-construction.

## Recommendation

The bones are good — JAX + Equinox + per-site `LatticeField` arithmetic + composable `Action` is exactly how to build this from scratch. But the codebase is carrying two parallel implementations of the field abstraction *and* the evolver, the fermion path is on the wrong side of the migration, and there are no tests to tell when something quietly breaks. Don't add gauge fields or more fermion machinery on top of the current state.

Concrete next ~week of work:

1. Add 6–10 invariant tests (a couple of hours; pays for itself immediately).
2. Migrate `Dirac4DFermionField` to the new `LatticeField` and drop the manual `_tree_flatten` registration.
3. Delete `OldLattice`/`OldLatticeField`/`HMCEvolver` once nothing imports them, or quarantine in `legacy.py`.
4. Fix the bugs above (`__iadd__`, `r_accept` indexing, `MRSolver` while-loop).
5. *Then* start on gauge fields — `WilsonDiracOp.shift_fermion` is already the right hook.

After that, the architecture earns the right to grow.
