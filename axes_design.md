# Internal axes: design decisions

Settled in discussion, 2026-11-23. This file records *design*; `arch_notes.md` records what
the code currently does. Where they differ, this is the target. It supersedes
`rewrite_thoughts.md` §3 (named internal indices).

**Guiding principle.** Make the operation that is wrong nine times out of ten the harder one
to reach for. Ambiguity is an error, never a silent default.

## Already in place (main, f952b79)

- `LatticeField.axes`: static `(name, size)` pairs after the spacetime axes. Names are unique
  per field for now.
- Arithmetic aligns internal axes by name. An axis on one side only broadcasts; disjoint
  axes give an outer product; one name with two sizes is an error.
- `.F` is the user-facing view (sites, then internal axes in declared order). Storage order is
  private and may change after benchmarking.
- Boundary conditions: exact phases or Dirichlet, with the product rule (see `arch_notes.md`).

## Now: axis operations and O(N)

### Reductions and products

All of these act per site. **None of them ever sums over sites**; totals belong to `Action`
and to observables. This is a deliberate difference from numpy (`np.vdot` flattens).

- `phi.contract(*names)`: Einstein summation over the named axes. Every occurrence of a name
  is set to one index and summed. With no names, every internal axis is contracted, giving a
  per-site scalar field. Boundary conditions are unchanged.
- `phi.dot(chi, *names)`: product, then contraction over the shared axes (all shared axes if
  no names are given). No conjugation, matching `np.dot`.
- `phi.vdot(chi, *names)`: the same with the first operand conjugated, matching
  `np.vdot`/`np.vecdot`. For real fields `dot` and `vdot` agree.

Because `dot` and `vdot` default to every shared axis, a term can be written without knowing
what the axis is called. For example, `phi.dot(phi.nn_field(mu))` works for any N and any
axis name, including N = 1.

### `lat.einsum(pattern, *operands, axes=None)`

This is the general engine. `contract`, `dot` and `vdot` are thin wrappers over it.

- **Syntax.** numpy's own single-letter syntax (`"ij,j->i"`), with two differences:
  - Spacetime is the implicit `...`. It is never written, is always elementwise, and the
    result is always a field. Users may not write `...` themselves.
  - `->` is required (no implicit output mode).
- **Operands.**
  - Fields must share a lattice.
  - Raw arrays are site-independent constants (γ, τ, ε, rotation matrices) and have no
    spacetime dimensions.
  - At least one operand must be a field.
- **Binding is per operand.** Labels are local to the call, as in numpy.
  - A *field's* labels bind to the axes listed in `axes=` that it has, in the order listed
    there. Its term must have one label for each such axis. With `axes=None`, a field binds
    all of its axes in `phi.axes` order.
  - A *raw array's* labels bind to all of its dimensions, in order.
- **Unlisted axes pass through.** They are aligned by name and elementwise, exactly like `*`,
  and are never summed: they behave like spacetime.
- **One label, one axis name.** A label bound to differently named axes on different fields
  is an error, even if the sizes match. A deliberate cross-space contraction goes through an
  explicit constant (a δ or embedding matrix).
- **Output naming.**
  - An output label bound to a field axis keeps that axis's name.
  - Labels carried only by constants fill the listed axes not already used by the output, in
    `axes=` order.
  - If that is ambiguous, or the counts don't match, it is an error asking for `axes=`.
- **Boundary conditions.** The product rule across all field operands. Raw arrays are
  neutral.

Examples:

```python
lat.einsum("a,a->", phi, chi)                         # φ·χ, a per-site scalar
lat.einsum("ij,j->i", gamma5, psi, axes="spin")       # color passes through
lat.einsum("ij,j->i", R, phi, axes="a")               # O(N) rotation
lat.einsum("abc,a,b,c->", eps, phi, dx, dy)           # O(3) topological charge density
lat.einsum("si,aij,sj,a->", psibar, tau, psi, phi,    # Yukawa: ψ̄ τ^a ψ φ_a
           axes=("spin", "flavor", "a"))
```

In the Yukawa example the binding table is:

| operand | listed axes it has | labels | binding |
|---|---|---|---|
| `psibar` | spin, flavor | `si` | s→spin, i→flavor |
| `tau` (raw) | — | `aij` | positional, sizes checked |
| `psi` | spin, flavor | `sj` | s→spin, j→flavor |
| `phi` | a | `a` | a→a |

There is no separate `apply(M, name)`: `einsum` with `axes=` covers it.

### Guards

- **`Action` rejects a density that still has internal axes.** The error points to
  `.contract()`. Without this, a scalar φ⁴ term applied to an O(N) field silently gives N
  decoupled φ⁴ fields, because the total sums every axis.
- **HMC momenta are `LatticeField`s** (via `copy_new_F`, keeping axes and BCs), and the kinetic
  density is `0.5 * p.dot(p)`. Today raw momenta of shape `(sites, N)` fail against a per-site
  H density. This is the minimal slice of the field-type protocol (`rewrite_thoughts.md` §2);
  updates on group or sphere manifolds and force projection wait for gauge fields.

### O(N) in `lfox.scalar`

O(N) extends `lfox.scalar`; the plain scalar is N = 1. There are no new public functions, and
about ten lines change.

- **Per-configuration observables** keep internal axes trailing. `magnetization` returns
  m_a, with shape `(N,)` (or `()` for a plain scalar).
  - `nn_product` uses `dot`.
  - `structure_factor` sums |·|² over components.
  - `correlator` sums only the transverse spacetime axes and contracts components in the
    product.
- **Ensemble estimators** contract the trailing component axis. Examples are
  χ = V(⟨m·m⟩ − ⟨m⟩·⟨m⟩) and Binder U = 1 − N/(N+2) · ⟨(m²)²⟩/⟨m²⟩²:
  - This U is 0 for a Gaussian and equals the usual 1 − ⟨m⁴⟩/3⟨m²⟩² at N = 1.
  - N is read from the column shape, so signatures don't change.
- **The O(N) action term is a test fixture, not a library preset.** `tests/phi4.py` is
  generalized, with the same density written with `dot`. The library gains no presets.
- **Physics tests:**
  - N = 1 reproduces the current φ⁴ term.
  - S(Rφ) = S(φ) for a random rotation R.
  - The autodiff force matches the analytic force.
  - ⟨exp(−ΔH)⟩ = 1.
  - At λ = 0, the result matches N copies of the exact free field (`tests/free_field.py`).
- **Out of scope:** the nonlinear sigma model (|φ| = 1). It needs the manifold field-type
  protocol, like gauge fields.

### Order of work

Failing tests come first at each step:

1. `einsum`, `contract`, `dot`, `vdot`.
2. The `Action` guard.
3. HMC momenta as fields.
4. Observables.
5. The O(N) term and physics tests.

## Later: matrix-valued fields (repeated axis names)

Decided, not to be implemented until a consumer exists (gauge links, most likely). Until then,
repeated names stay an error at construction.

A matrix on one space carries **the same axis name twice**: U has `(color, color)`, Γ has
`(spin, spin)`. This mirrors Einstein notation.

- **Same-named axes** must have the same size (they are one space). Their stored order is
  meaningful: the first is the row, the second the column. Any future change of storage order
  may reorder different names, but must keep same-named axes in their relative order.
- **Alignment rule.** A name present in both operands must occur the same number of times in
  each; occurrences pair up in stored order. A name present on one side only broadcasts, as
  now. So `U + psi` and `U.dot(psi)` are errors: ψ's one color axis could pair with either of
  U's.
- **`*` and `/` are errors when both operands carry a repeated name.**
  - Elementwise (Hadamard) products of matrices are almost never meant. Plaquette code written
    as `U_mu * U_nu.nn_field(mu)` must not silently compute U_ij V_ij.
  - The error points to `@`. A genuine elementwise product is spelled
    `einsum("ij,ij->ij", ...)`.
  - Products that share no repeated name are fine. Scalar fields times matrices are fine, and
    `Gamma * U` is a tensor product.
- **A raw scalar plus a field with a repeated name is an error**, since `1 + U` almost always
  means the identity. The identity is spelled explicitly with a `lat.eye` helper (agreed), or
  einsum. *Constructing* a field with axes from a scalar (`F=0.0`) remains fine.
- **`contract(name)` is Einstein summation:** every occurrence is set to one index and summed.
  - On a pair, that is the trace: `U.contract("color")` = Tr U.
- **`vdot`** pairs shared axes in stored order:
  - `U.vdot(V)` = Σ conj(U_ij) V_ij = Tr(U†V), the Frobenius inner product. The gauge
    kinetic term is `P.vdot(P)` = Tr P².
  - **Trap to document:** `U.dot(V)` is Tr(UᵀV), not Tr(UV). The latter is
    `(U @ V).contract("color")`.
- **`@`:** for every name present in both operands, contract the left operand's *last*
  occurrence with the right operand's *first*.
  - This covers U @ V, U @ ψ, ψ̄ @ U and `(Gamma * U) @ psi` (spin and color at once).
  - For vectors, φ @ χ is `dot`, as in numpy.
  - A gauge transformation reads `Om @ U @ Om.nn_field(mu).H`.
- **`.T`** reverses the order within each repeated name; **`.H`** is `.conj().T`.
- **`einsum`:**
  - Same-named axes bind in stored order. For a τ field with `(flavor, flavor, a)` and
    `axes=("spin", "flavor", "a")`, its term is `"ija"`, and `"aij"` is caught by the
    one-label-one-name rule.
  - Listing a name in `axes=` binds every occurrence.
  - An output may repeat a name, in the order written.
- **Blind spot.** U and Uᵀ have the same type, so a row/column mix-up (`"ij"` vs `"ji"`) is
  invisible, as in numpy. Covariance tests are what catch it.
- **Rectangular maps** between different spaces (N×M flavor mixing) use two different names;
  nothing new is needed.

## Open questions

- **Hop directions vs array axes.** On the honeycomb lattice the unit-cell axis is an array
  axis but not a hop direction, and `d()` currently counts it. `Lattice` should own its hop
  directions.
- **Gauge links:** one field with a `mu` axis, `(mu, color, color)`, or one field per
  direction? A `mu` axis interacts with `nn_field(mu)`, since it indexes hop directions, not
  an internal space. Deliberately deferred until link fields are being implemented: it is an
  important decision, but it can only be properly confronted with a real consumer.
- Storage-order benchmark (sites-leading vs internal-leading).
- Port `fermions/spin.py` from `indices` to `axes`, and its bilinears to `lat.einsum`.
- Dirichlet + periodic in `+` is an error; it might become a warning if a real use appears.
- Out of scope: G-parity, C* and 't Hooft twisted BCs, and Schrödinger-functional boundary
  layers.
