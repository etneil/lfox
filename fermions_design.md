# Fermions: design decisions

Settled in discussion, 2026-11-23, on branch `fermions`. This file records *design*;
`arch_notes.md` records what the code currently does. It replaces the fermion plans in
`rewrite_thoughts.md` §3 and `PLAN.md` step 4.

Scope: the machinery for fermions in HMC (spinors, gamma matrices, the Wilson operator,
a solver with diagnostics, pseudofermions), pinned by unit tests. The full Yukawa
integration test is the user's, and comes after.

## What a fermion field is

A spinor is a plain `LatticeField` with a spin axis, usually complex, usually
antiperiodic in time: `LatticeField(F=..., axes={"spin": 4}, bc=(-1, 1, 1, 1))`.

- There is no `Dirac4DFermionField` subclass. Its `bilinear`, `dot` and `norm_sq` are now
  `vdot` and `lat.einsum`; antiperiodic time and the product rule come from the boundary
  conditions (σψ is antiperiodic automatically); color and flavor are more axes.
- In HMC a fermion appears only as a pseudofermion, a complex *sampled* field, and through
  the operator D acting on such fields. Nothing is Grassmann.

## Gamma matrices (`lfox.fermions`)

Euclidean, Hermitian, {γ_μ, γ_ν} = 2δ_μν, for d = 2, 3, 4, plus γ5 in even d:

- d = 4: DeGrand–Rossi (the current basis), γ5 = γ0 γ1 γ2 γ3 with the sign fixed by it;
- d = 2: Pauli matrices, γ5 = −iγ0γ1 = σ3 (spin size 2);
- d = 3: Pauli matrices (spin size 2; no γ5).

## Operators are linear functions of fields

A Dirac operator is a plain linear function of the fermion field, closed over its bosonic
background:

```python
D = lambda psi: wilson(psi, kappa) + g * sigma * psi    # a Yukawa coupling is just a term
```

- **Wilson, hopping form:**
  D ψ = ψ − κ Σ_μ [(1 − γ_μ) ψ(x+μ̂) + (1 + γ_μ) ψ(x−μ̂)].
  With lfox's `nn_field` convention (site x receives x − shift), ψ(x+μ̂) is
  `psi.nn_field(mu, -1)`. The free operator on a plane wave is
  1 − 2κ Σ_μ (cos p_μ − iγ_μ sin p_μ); antiperiodic time shifts p_0 by π/L_0.
  Gauge links later enter through the hop (the old `shift_fermion` hook).
- **Adjoint, for any linear operator:** `dagger(D, like)` via `jax.linear_transpose` plus
  conjugation. γ5-hermiticity (γ5 D γ5 = D†) holds for Wilson but not for every operator we
  want: a pseudoscalar Yukawa coupling iγ5π breaks it. So γ5-hermiticity is a *test* of the
  Wilson operator, not the implementation of `dagger`.
- **Dense matrix, for small lattices:** `dense(D, like)` builds the matrix of a linear
  operator by applying it to basis vectors. It gives exact determinants and spectra for
  tests, and exact checks of pseudofermion HMC on tiny lattices.

`dagger`, `dense` and the solver are not fermion-specific; they live in `lfox.linalg`.

## Solver: CG with diagnostics (`lfox.linalg.cg`)

`cg(A, b, x0=None, tol=..., maxiter=...) -> (x, info)` for Hermitian positive-definite A
(M†M), with `info = {"iterations", "residual", "converged"}`.

- Our own CG, about 20 lines: a `lax.while_loop` wrapped in `lax.custom_linear_solve` with
  `has_aux=True`. So it is jittable, its diagnostics come out, and it differentiates
  implicitly (one extra solve in the backward pass, never through the iterations).
- **Gotcha, pinned by a test:** for complex Hermitian A, Aᵀ = conj(A) ≠ A, so the solve must
  be declared with `transpose_solve=solve`, not `symmetric=True`. `symmetric=True` gives a
  silently wrong gradient (found by probe: 89.7 against a finite-difference 96.97).
- `jax.scipy.sparse.linalg.cg` was rejected only because it reports no iteration count.
- M⁻¹b, for propagators, is (M†M)⁻¹M†b. A BiCGStab for non-Hermitian M can come later.

## Diagnostics: from the solve to the HMC monitor

Solves run inside `Term.total`, inside `jax.grad` and the integrator loop. Their
diagnostics flow out through one channel:

- **Terms:** `total` may return `(value, info)` with `info` a dict. Terms without solves are
  unchanged. A pseudofermion term writes `x, info = cg(...); return S, info`.
- **`Action`:** `S` still returns a number. A variant returns `(S, {term name: info})`, and
  `dS` can return info alongside the gradient (`jax.value_and_grad(..., has_aux=True)`).
- **Integrators:** `delta_P` returns `(dP, info)`. Leapfrog and Omelyan move from
  `fori_loop` to `lax.scan`, which stacks the info of every force evaluation.
- **HMC monitor:** every solve of a trajectory, in order (the accept/reject evaluations and
  each force evaluation), stacked under flat keys `"<term>.<key>"`, for example
  `monitor["Pseudofermion.iterations"]` with shape `(n_solves,)` per trajectory and
  `(traj, n_solves)` from `evolve_many`. Stacking means no reduction rule (sum, max) is
  baked in.
- **Names:** info is keyed by term name, so two reporting terms with one name (on different
  fields) are an error asking for labels. `replicate` suffixes the labels of the copies it
  makes (`Pseudofermion_0`, `Pseudofermion_1`, ...), but only of terms on a replicated field;
  a term on none of them is still a refused duplicate.
- **Hand-written gradients:** a term with a `gradient` hook is never differentiated, so its
  `dS` info comes from `gradient`, which may return `(grads, info)` by the same convention.
- **delta_H from totals:** H = S + Σ ½p·p, so actions with total-only terms run. Differencing
  two O(V) totals loses ~1e-7·V in float32 (per-site differencing kept it near √V·1e-7). This
  is standard practice: production codes evaluate the Metropolis H in double precision even
  when the MD runs lower, and an accurate pseudofermion S needs a double-precision solve
  anyway. Future work: a precision setting for the Metropolis evaluations (upcast to float64
  for the two H evaluations, float32 MD), and mixed-precision solves (iterative refinement:
  double-precision residual, low-precision inner `cg`) as a wrapper around `cg`. Nothing in
  the plumbing assumes one precision.
- **Integrators:** `Nstep` is static (the monitor's shape depends on it). Each step is a
  self-contained scan body (drift-kick-drift for leapfrog), so trajectories agree with the
  old `fori_loop` form only to rounding.
- **Non-convergence:** recorded, never forced into a rejection (that would hide a solver
  problem behind a lower acceptance). `Chain` warns, host-side, when any monitor entry
  `"<term>.converged"` in a block is false.
- **Known limitations:** differentiating through `cg` adds a backward-pass (adjoint) solve
  whose diagnostics JAX discards. `PseudoFermion`'s force avoids it (`stationary_form`,
  below), so its monitor entries count every solve.

## Pseudofermions

- `total` = η†(M†M)⁻¹η, a nonlocal term returning `(S, info)`.
- `draw`: η = M†ξ with ξ a complex Gaussian of density exp(−ξ†ξ) (ξ = (a + ib)/√2); then
  η†(M†M)⁻¹η = ξ†ξ. The term declares `sampled = ("eta",)`.
- **The force is one solve, by `lfox.linalg.stationary_form(A, b)`**: b†A⁻¹b computed as
  max_x [2 Re(x†b) − x†Ax] at x = A⁻¹b, with x frozen by `stop_gradient`. Stationarity in x
  makes first derivatives exact without differentiating the solve (differentiating
  `vdot(b, cg(A, b))` costs a second, adjoint solve). Second derivatives are not exact,
  which matters for force-gradient integrators; documented, not guarded. Measured at L = 4:
  36 µs against 66 µs per force. (The value is no more accurate than `vdot(b, x)`: for CG
  from x0 = 0 the two coincide by Galerkin orthogonality. Its one guarantee on the value is
  that it never exceeds b†A⁻¹b, for any x.)
- Two degenerate flavors fit one pseudofermion of M†M; more flavors are more pseudofermions
  (`replicate(n, "eta")`). An odd number needs RHMC: out of scope.
- **Library class `lfox.fermions.PseudoFermion`**, promoted from the test fixture once two
  integration notebooks had each rewritten it. A subclass supplies only
  `fermion_matrix(self, *bosons)`, returning M as a linear function of spinors, and gets
  `total` (plain `cg`, exact to all orders), `draw` (η = M†ξ) and `gradient` (via
  `stationary_form`, covering every role). The name is `fermion_matrix`, not "Dirac
  operator", because the class needs only linearity: non-relativistic (Hubbard, nuclear
  EFT) fermion matrices fit too.
  - Roles: `fermion_matrix`'s parameters, then `eta`, declared from `__init_subclass__`
    through `Term.roles` (a base class whose hooks are generic over its subclasses' fields
    declares them; its hooks take `**fields`). This keeps gauge (`fermion_matrix(self, U)`)
    and gauge-Yukawa (`fermion_matrix(self, U, phi)`) theories in reach.
  - Guards: `fermion_matrix` must name its fields and may not take `eta`; a subclass
    overriding `total` must override `gradient` too (the inherited force would belong to
    the base action). Overriding `gradient` alone, with a hand-written force, is the
    intended optimization path.
  - Later: an `ExactFermion` sibling sharing `fermion_matrix`, with S = −log det(M†M) by
    `dense`, for exact cross-checks at small volume. Hasenbusch and rational terms would be
    siblings built on `stationary_form`, not subclasses.
  - For gauge fields: `dS` returns the plain gradient; projection onto the Lie algebra
    belongs to the group-aware evolver. The first evolved complex field must also pin
    which Wirtinger derivative `dS` returns.

## Legacy code

`Dirac4DFermionField`, `MRSolver` and the current `wilson.py` are deleted and rewritten,
not ported: none of it runs against the current `LatticeField`.

## Order of work

Failing tests first at each step, pausing for review.

1. Gamma matrices, `wilson`, `dagger`, `dense`: Clifford algebra; free plane-wave spectrum,
   including antiperiodic time; γ5-hermiticity of Wilson; `dagger` agreeing with γ5 D γ5.
2. `cg`: converges on known systems; iteration count and `converged`; `maxiter` stops a
   non-converging solve; complex-Hermitian gradient against finite differences; jit.
3. Diagnostics plumbing: a toy term returning info; monitor keys and shapes for both
   integrators; `Chain` warns on non-convergence; nothing changes for terms without info.
4. Pseudofermion fixture (`tests/pseudofermion.py`), unit-tested against the dense M†M:
   the action; the heatbath covariance ⟨ηη†⟩ = M†M, ⟨ηηᵀ⟩ = 0 (stronger than equipartition);
   the force against dense autodiff and finite differences. The statistical HMC checks
   (⟨e^{−ΔH}⟩ = 1, sampling with weight det(M†M)) are integration-scale and left to a
   scalar-Yukawa integration test.
5. `stationary_form` and the library `PseudoFermion` (above), with `Term.roles`; the fixture
   becomes a subclass (`YukawaFermion`). HMC-level checks live in
   `integration/reproduce_kanwar.py`: ⟨e^{−ΔH}⟩ = 1, and pseudofermion HMC against HMC with
   the exact determinant.
