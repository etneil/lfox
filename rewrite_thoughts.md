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
- `Lattice.__eq__` (`lattice.py:80-81`) raises `NotImplementedError` instead of returning the `NotImplemented` singleton. `lattice == 5` explodes rather than returning `False`. One-line fix.
- `LatticeField.copy_new_F` (`lattice.py:288-294`) — the `cls.__new__` + `__dict__.update` + `dataclasses.replace` dance is more contortion than needed. `dataclasses.replace(self, F=new_F)` alone should work since `LatticeField` is dataclass-flavored.
- Dead commented-out alternate `nn_field` block at `lattice.py:254-279` — delete.

## Recommendation

The bones are good — JAX + Equinox + per-site `LatticeField` arithmetic + composable `Action` is exactly how to build this from scratch. But the codebase is carrying two parallel implementations of the field abstraction *and* the evolver, the fermion path is on the wrong side of the migration, and there are no tests to tell when something quietly breaks. Don't add gauge fields or more fermion machinery on top of the current state.

Concrete next ~week of work:

1. Add 6–10 invariant tests (a couple of hours; pays for itself immediately).
2. Migrate `Dirac4DFermionField` to the new `LatticeField` and drop the manual `_tree_flatten` registration.
3. Delete `OldLattice`/`OldLatticeField`/`HMCEvolver` once nothing imports them, or quarantine in `legacy.py`.
4. Fix the bugs above (`__iadd__`, `r_accept` indexing, `MRSolver` while-loop).
5. *Then* start on gauge fields — `WilsonDiracOp.shift_fermion` is already the right hook.

After that, the architecture earns the right to grow.

## Forward-looking design proposals

Two design changes worth making *before* the next round of physics, since both touch APIs that will harden as more code is written on top.

### Named internal indices

Replace the positional `indices: tuple[int]` on `LatticeField` with named, ordered axes:

```python
class LatticeField(eqx.Module):
    lattice: Lattice = eqx.field(static=True)
    F: jax.Array
    bc: tuple[int, ...] = eqx.field(static=True)
    # ordered, named internal axes following the spacetime axes in F
    axes: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())

    def axis_pos(self, name: str) -> int:  # static, called at trace time
        d = len(self.lattice._dims)
        for i, (n, _) in enumerate(self.axes):
            if n == name:
                return d + i
        raise KeyError(name)
```

Operators target axes by name but compile to positional ops:

```python
@jax.jit
def gamma_apply(field, gamma_mat):
    pos = field.axis_pos('spin')          # static, resolved at trace
    return field.copy_new_F(
        jnp.tensordot(gamma_mat, field.F, axes=[[1], [pos]]).moveaxis(0, pos)
    )
```

**Why it doesn't cost performance:** the name→position map is `static=True`, so `axis_pos` runs in Python during tracing and disappears from the compiled HLO. The jaxpr is byte-identical to a hand-written positional einsum (verifiable with `jax.make_jaxpr`). The one rule: never build an einsum string from a runtime value inside a jit — since names live on the static field type, that won't happen by accident.

**Why it composes well with sharding:** you partition spacetime axes via `Mesh` + `NamedSharding`/`PartitionSpec`; internal indices (spin/color) are typically replicated and contracted locally. Named internal axes makes that boundary explicit and self-documenting. If color ever needs sharding (very large N), it's a one-line spec change because the axis name maps to a known static position.

**Why now:** `Dirac4DFermionField._inner_product` already hardcodes `'...i,...i'` assuming spin is the only trailing axis. The moment a gauged Dirac op adds a color index, every einsum in the fermion code needs to remember positional order. Named axes prevents a class of silent contraction bugs before they get written.

### Per-term action params (drop the flat-merge)

`Action.add_subaction` currently merges sub-action `params` into a single flat dict (last-write-wins, TODO flagged). Replace with per-sub-action params: each `Action` carries its own `params`, and `S_field` walks sub-actions calling each with its own slice.

Authoring ergonomics are preserved verbatim:

```python
S = GaugeAction(field_names=['U'], params={'beta': 5.0})
S = S + WilsonAction(field_names=['psi'], params={'kappa': 0.13})
S = S + ScalarAction(field_names=['phi'], params={'lam': 0.1, 'm2': -2.0})
```

The case the flat dict can't handle today falls out for free:

```python
psi1 = WilsonAction(field_names=['psi_1'], params={'kappa': 0.13})
psi2 = WilsonAction(field_names=['psi_2'], params={'kappa': 0.14})
S = gauge_act + psi1 + psi2   # two κ's coexist; flat-merge would clobber
```

**Pair this with committing to dicts on the field side**, and handle many-flavor aliasing with an explicit rename map instead of positional list-vs-dict ordering:

```python
psi2_act = WilsonAction(field_names=['psi'], params={'kappa': 0.14}) \
            .rebind({'psi': 'psi_2'})
```

`rebind` returns a new action whose `S_field` does `fields[rebind_map.get(name, name)]` — one indirection at trace time, zero at runtime. The list-vs-dict hybrid goes away; the multi-flavor motivation is satisfied; `_S`, `S`, `S_field` all take dicts.

**User-visible change:** `S.params['kappa']` from outside is gone. Replace with `S.with_param('kappa', 0.14)` returning a new action (pure-functional, JIT-safe; sub-action targeting via path). Notebooks that mutate `params` in place need updating — but those mutations already conflict with `params` being `static=True`, so they're fragile today.

**Why now:** with gauge β, Wilson κ, pseudofermion mass, and eventually multi-flavor all landing at once, the flat-merge collision is going to bite during the gauge-field push. Cleaner to remove the foot-gun first.

### Open question to resolve before gauge fields

How does a `LinkField` (or `geometry='link'` tag on `LatticeField`) interact with the integrator? Sketch:

- Each field type knows how to advance itself given a tangent-space momentum: `field.advance(p, dt)`. Scalars use `+`; group-valued links use `U → exp(i·dt·P)·U`.
- `MDIntegrator.update` calls `field.advance(...)` instead of `+`, dispatching per-field.
- Forces from `jax.grad` need to land in the algebra, not the group — either parameterize-and-exponentiate (cleanest for autodiff) or carry-and-project. Decide before the first link-valued action lands.

If this gets sketched before fermions get heavier, scalar HMC remains a clean special case. If not, expect a parallel `GaugeHMC` evolver to appear and the same `HMCEvolver` vs `HMCRewrite` bifurcation to repeat one level up.
