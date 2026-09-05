# lfox rewrite thoughts

Revised 2026-09-04 against commit `86d5c8b`. Supersedes the May 2026 memo; the
detail in `PLAN.md` steps 3-4 is superseded by this file too (steps 1-2 there are done).

Bottom line up front: the foundation has converged. The legacy `OldLattice` /
`OldLatticeField` / `HMCEvolver` code is gone, there is one pure `HMC` kernel driven
by a host-side `Chain`, and 32 tests pin the scalar physics. What remains is
concentrated in three places:

1. **`Action` composition and params are buggy and untested.** Three reproducible
   bugs, one of which silently returns wrong physics.
2. **`lfox.fermions` is broken end to end** and should be rewritten, not patched.
3. **HMC has no notion of a field type.** Momenta, kinetic energy and the update step
   are hard-wired for scalars. Gauge fields cannot land until that is a protocol.

Do those three in that order, then gauge fields. Everything else is small.

## Done since the May memo

For the record, so nobody re-investigates:

- `Action.__iadd__` returns `self`; `__copy__` is clean; the `r_accept` off-by-one and
  the field-mutating `HMCEvolver` died with the old evolver.
- `OldLattice` / `OldLatticeField` deleted. `HMCEvolver` vs `HMCRewrite` resolved into
  `HMC` (pure kernel, `lax.scan` in `evolve_many`) plus `Chain` (stateful driver).
- `Lattice.__eq__` resolved by removal: the precomputed `_bc_coords` grid is gone,
  `bc_coord(axis)` is computed on demand, and default dataclass equality/hash works.
  `lattice == 5` returns `False`. (The architecture note in `CLAUDE.md` still describes
  `_bc_coords` and an overridden `__eq__`/`__hash__`; that paragraph is stale.)
- Dead commented-out `nn_field` block deleted.
- Tests exist: `tests/test_lattice.py` (BC winding), `tests/test_scalar_hmc.py`
  (action normalization, autodiff vs analytic force, RNG threading, `<exp(-dH)> = 1`),
  `tests/test_chain.py` (bookkeeping, blocking invariance, seed reproducibility).
- Notebooks thinned; jupytext pairing; the Schaefer reproduction lives in `integration/`.

## Real bugs (all reproduced 2026-09-04)

### `Action` composition and params

None of these are covered by a test. `PLAN.md` lists "Action composition: `+`, `+=`,
nested `sub_actions`, `params` merging" as a test to write; write it first, it fails today.

- **`a + b` mutates `a.params`.** `__copy__` (`action.py:111-113`) does
  `dataclasses.replace(self, sub_actions=...)`, which shares the `params` dict, and
  `add_subaction` then updates it in place (`action.py:121`). After `c = a + b`,
  `a.params == c.params`.
- **`a + (b + c)` raises `FrozenInstanceError`.** `add_subaction` assigns
  `other.sub_actions = []` (`action.py:128`) on a frozen `eqx.Module`. Nested
  composition does not work at all.
- **In-place mutation of `params` poisons the jit cache.** `params` is a static field, so
  the dict *object* is part of the jit cache key. Mutating it rewrites the key under the
  already-compiled executable. Even a brand-new action with the new value then hits the
  stale entry:

  | call                                                | result  | expected |
  |-----------------------------------------------------|---------|----------|
  | `act.S(fields)` with kappa = 0.18                   | 271.164 | 271.164  |
  | `act.params['kappa'] = 0.0`; `act.S(fields)`        | 271.164 | 270.007  |
  | fresh `ScalarAction(params={'kappa': 0.0, ...}).S`  | 271.164 | 270.007  |

  Third row is silent wrong physics from an action that was never mutated.
- **Sub-actions receive the parent's merged params**, not their own
  (`action.py:72`: `action._S(fields=..., params=self.params)`). Two Wilson terms with
  different kappa cannot coexist, which is exactly the multi-flavor case the list-of-fields
  convention was meant to enable.

### `lfox.fermions` (`import lfox.fermions.spin` raises)

- `spin.py:107-111` registers `_tree_flatten` / `_tree_unflatten`, which do not exist on
  `LatticeField` (`AttributeError` at import).
- `spin.py:65` defines a custom `__init__`, which suppresses `LatticeField.__post_init__`:
  no default boundary conditions, no scalar broadcast. Equinox warns about this at import.
- `spin.py:76,78,83`: `bilinear` and `conj` assign to `.F` on a frozen module.
- `wilson.py:23` calls `zero_fill()`, which does not exist; `wilson.py:25` uses
  `psi.lattice.d`, which does not exist (`d()` is a method on `LatticeField`).
- `solver.py:16-19`: `iter` is initialized and never incremented, so `max_iter` is dead,
  a non-converging solve loops forever, and the returned iteration count is always 0.
  The loop is also host-side Python; it should be `jax.lax.while_loop` with a residual
  predicate.
- `WilsonDiracOp` and `MRSolver` are plain classes jitted with `static_argnums=(0,)`,
  so they hash by identity and every new instance recompiles.

This is ~180 lines. Rewrite against the current `LatticeField`, tests first.

### Smaller

- `lattice.py:96`: `type(self.F) != jnp.array` compares against a function, so it is
  always `True`. The broadcast-multiply always runs; an `int32` field is silently
  upcast to `float64`. Use `self.F.ndim == 0` or `jnp.broadcast_to`.
- `lattice.py:102`: the `type(self.lattice) is dict` workaround is dead.
- `lattice.py:195-201`: `copy_new_F` still does the `__new__` + `__dict__.update` dance.
  Both `dataclasses.replace(self, F=F)` and `eqx.tree_at(lambda f: f.F, self, F)` are
  verified to work. `tree_at` also skips re-running `__post_init__` on every arithmetic op.
- `lattice.py:75`: `HoneycombLattice.shift` FIXME stands. `rb_split` / `rb_combine`
  (`lattice.py:40-51`) are stubs that return `None`. Delete or mark unsupported.
- `integrators.py:35,64`: the inner `jax.jit(static_argnums=(1, 2))` on `_integrate`,
  with the force closures as static args, is noise. `HMC.evolve` is the jit boundary and
  repeated `evolve` calls hit its cache; the inner jit adds nothing under it and would
  recompile on every call outside it. Remove it and keep one boundary.
- `integrators.py:9-10`: `eps` and `Nstep` are pytree leaves, so they are *traced* inside
  `evolve`. Consequence: changing either does not retrace (nice for tuning), but
  `fori_loop` lowers to a `while_loop` with an unknown trip count. Make a deliberate
  choice; suggest `Nstep` static, `eps` traced.
- `hmc.py:40-46`: `momentum_refresh` returns raw arrays, which only become
  `LatticeField`s after the first integrator `update`. Works by accident; fixed for free
  by proposal 2 below.

## Architectural calls that held up

Unchanged from May, now confirmed by a year of churn:

- `eqx.Module` for `Lattice` / `LatticeField` / `Action` / integrators / `Evolver`, with
  `Lattice` as a static field on `LatticeField`.
- `_nn_field` as a jitted staticmethod with everything it needs as static args.
- Composable `Action` via `sub_actions` and `+` (the *shape* is right; the params
  plumbing is what is broken).
- Pure `Evolver` + `lax.scan` in `evolve_many`; stateful `Chain` entirely outside jit.
  The pure/stateful boundary is the jit boundary. Keep it that way.

## Design proposals

### 1. Per-term, immutable params; dicts everywhere; `rebind` for aliasing

Each `Action` carries its own `params`; `S_field` walks sub-actions calling each with its
own slice. No flat merge, no `params.update`. Authoring is unchanged:

```python
S = GaugeAction(field_names=['U'], params={'beta': 5.0})
S = S + WilsonAction(field_names=['psi_1'], params={'kappa': 0.13})
S = S + WilsonAction(field_names=['psi_2'], params={'kappa': 0.14})   # two kappas coexist
```

Commit to dicts at every layer (`_S`, `S_field`, `S`, `dS` all take `{name: field}`),
and handle many-flavor aliasing with an explicit rename instead of positional lists:

```python
psi2 = WilsonAction(field_names=['psi'], params={'kappa': 0.14}).rebind({'psi': 'psi_2'})
```

`rebind` returns a new action whose `S_field` looks up `fields[rebind_map.get(name, name)]`.
One indirection at trace time, zero at runtime.

Post-construction mutation goes away. Replace `S.params['kappa'] = x` with
`S.with_param('kappa', x)` returning a new action.

**New option, not in the May memo: make `params` a pytree leaf, not static.** The
staticmethod `_S` already receives `params` as a traced argument, so tracing does not
change. Benefits: a coupling scan does not retrace; cache poisoning becomes impossible by
construction; `with_param` is `eqx.tree_at`. Cost: params must be JAX-typed scalars
(floats/ints are fine; no strings or shape-determining ints in `params`). Recommend
trying non-static first and falling back to static-plus-frozen if something needs a
Python-level parameter.

**Decision needed:** what observables receive. `Chain._record_step` currently passes
`evolver.action.params` (the flat dict) to `obs_f(fields, params)`. Options: pass the
action itself, or expose a merged read-only view. Passing the action is simplest and
lets observables call `action.S_field` directly.

### 2. A field-type protocol for HMC

Today HMC and the integrators assume scalars in three places: `momentum_refresh`
(`jax.random.normal` with `F.shape`), `H_density` (`0.5 * pi**2`), and
`MDIntegrator.update` (`X + dt * P`). Group-valued links need Lie-algebra momenta with a
different shape from `U`, a trace for the kinetic term, and `U -> exp(i dt P) U` for the
update. Put those three on the field type:

```python
class LatticeField(eqx.Module):
    def random_momentum(self, key) -> "LatticeField": ...   # Gaussian, same shape as F
    def kinetic(self, p) -> "LatticeField": ...            # 0.5 * p**2, per site
    def advance(self, p, dt) -> "LatticeField": ...        # self + dt * p

class LinkField(LatticeField):
    def random_momentum(self, key): ...   # algebra-valued, shape (..., N, N) traceless anti-Hermitian
    def kinetic(self, p): ...             # 0.5 * sum_a p_a**2 per site; normalization must match random_momentum
    def advance(self, p, dt): ...         # expm(dt * p) @ self.F
```

`HMC.momentum_refresh`, `HMC.H_density` and `MDIntegrator.update` dispatch to these and
carry zero knowledge of field types. Momenta are `LatticeField`s from the start, which
also fixes the raw-array-then-`LatticeField` inconsistency noted above.

**Forces:** commit to carry-and-project. `jax.grad(S)` with respect to the matrix entries
of `U`, then project `U dS/dU^dagger` (or the equivalent) onto the traceless
anti-Hermitian part. This composes with autodiff for free and is what the existing JAX
lattice codes do. `delta_P` becomes `field.project_force(grad)` with the scalar
implementation being the identity. Parameterize-and-exponentiate is equivalent at first
order but makes the parameterization trajectory-local; not worth it.

**Why now, before fermions:** it needs no gauge fields to exist. Do it on scalars against
the existing `<exp(-dH)> = 1` test plus a new leapfrog reversibility test, and scalar HMC
stays a clean special case. If this waits until a `LinkField` exists, expect a parallel
`GaugeHMC` and the same bifurcation the old `HMCEvolver` / `HMCRewrite` split had.

### 3. Named internal indices

Replace positional `indices: tuple[int]` with ordered, named axes:

```python
class LatticeField(eqx.Module):
    lattice: Lattice = eqx.field(static=True)
    F: jax.Array
    bc: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())   # after spacetime

    def axis_pos(self, name: str) -> int:   # static, resolved at trace time
        ...
```

Operators target axes by name but compile to positional ops; the name-to-position map is
static, so the jaxpr is identical to a hand-written positional einsum. The one rule: never
build an einsum string from a runtime value.

**Timing:** the only consumer of `indices` today is the broken fermion code, so this is
the cheapest it will ever be, but do it *as part of* the fermion rewrite rather than
before it, so there is a consumer to test against. The payoff is largest once gauge
fields exist: links carry `(mu, color, color)`, fermions `(spin, color)`, and positional
einsum strings across the two types is where silent contraction bugs would live.

## Ordered work

Tests first at every step (see `CLAUDE.md`). Each step lists the tests that should fail
before the code changes and pass after.

### 1. `Action` composition and params

Tests (new file `tests/test_action.py`):
- `a + b` leaves `a` unchanged (params and `sub_actions`).
- `a + (b + c)` works and flattens to three terms.
- `(a + b).S == a.S + b.S` on a random field.
- Two sub-actions with the same param name and different values give the right total.
- `with_param` / `rebind` return new actions; the original is unchanged; jit results are
  correct for both (guards against cache poisoning).
- `_S` receives a dict; a two-flavor action via `rebind` evaluates correctly.

Then: fix the three bugs, implement proposal 1, update `Chain` observables, update
`tests/phi4.py` and `integration/reproduce_schaefer.py` to the dict convention.

### 2. Field-type protocol for HMC

Tests:
- Leapfrog and Omelyan reversibility: integrate forward, negate momenta, integrate back,
  recover `X` to ~1e-10 in float64.
- Momenta returned by `momentum_refresh` are `LatticeField`s with the field's lattice/bc.
- Existing `<exp(-dH)> = 1` and detailed-balance tests unchanged.

Then: add `random_momentum` / `kinetic` / `advance` / `project_force` to `LatticeField`,
dispatch from `HMC` and `MDIntegrator.update`. Remove the inner jit on `_integrate`; make
`Nstep` static.

### 3. Fermion rewrite with named indices

Tests (new `tests/test_spin.py`, `tests/test_wilson.py`, `tests/test_solver.py`):
- Gamma algebra: `{g_mu, g_nu} = 2 delta_mu_nu`, Hermiticity, `g5**2 = 1`, and
  `g5 = g0 g1 g2 g3` with the sign fixed by the DeGrand-Rossi definitions in `spin.py`.
- `Dirac4DFermionField` construction gets default BCs; `bilinear` / `conj` return new fields.
- Wilson operator: `g5`-Hermiticity (`g5 D g5 = D^dagger`); on a plane wave the free
  operator reduces to `1 - 2 kappa sum_mu (cos p_mu - i g_mu sin p_mu)` acting on the
  spinor, checked on a small lattice; antiperiodic time BC shifts the allowed `p_0` by
  `pi / L_0`.
- MR solver: converges on a known small matrix to the requested residual; returned
  iteration count is correct; `max_iter` terminates a non-converging solve.

Then: rewrite `spin.py` / `wilson.py` / `solver.py` against the current `LatticeField`
with `axes=(('spin', 4),)`, make `WilsonDiracOp` and `MRSolver` `eqx.Module`s,
`lax.while_loop` in the solver, drop the pytree registration.

### 4. Small cleanups

Fold into whichever of 1-3 touches the file: `lattice.py:96` type check,
`lattice.py:102` dead branch, `copy_new_F` via `tree_at`, honeycomb FIXME and rb stubs.
Flag the stale `CLAUDE.md` architecture paragraph for a human edit.

### 5. Gauge fields

Only after 1-3: `LinkField` implementing the protocol from step 2, a Wilson plaquette
action, `WilsonDiracOp.shift_fermion` overridden to multiply by the link. Tests: gauge
invariance of the plaquette action under a random gauge transformation, force vs.
finite difference, `<exp(-dH)> = 1` for pure gauge on a tiny lattice, plaquette
expectation value against a known strong-coupling or literature number.

## Reproducing the cache-poisoning bug

```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import lfox.lattice as lat
from tests.phi4 import ScalarAction

L = lat.SquareLattice(st_dims=(4, 4, 4))
fields = {'phi': lat.LatticeField(lattice=L, F=jax.random.normal(jax.random.PRNGKey(0), (4, 4, 4)))}

act = ScalarAction(field_names=['phi'], params={'kappa': 0.18, 'lambda': 1.3})
print(act.S(fields))                       # 271.164
act.params['kappa'] = 0.0
print(act.S(fields))                       # 271.164, expected 270.007
fresh = ScalarAction(field_names=['phi'], params={'kappa': 0.0, 'lambda': 1.3})
print(fresh.S(fields))                     # 271.164, expected 270.007 (!)
```

Run with `PYTHONPATH=. uv run python <file>` from the repo root.
