"""Design tests for operations on internal axes, written before the implementation.

The design is recorded in `axes_design.md` ("Now: axis operations and O(N)").
Everything here acts *per site*: nothing ever sums over the lattice, so every
result is again a `LatticeField` on the same lattice.  Totals belong to
`Action` and to observables.

  * **`lat.einsum(pattern, *operands, axes=None, out_axes=None)`** is numpy's
    einsum, site by site:
      - numpy's single-letter syntax.  Spacetime is the implicit `...`: never
        written (writing `...` is an error), always elementwise.  `->` is
        required.
      - operands are fields on one lattice, or raw arrays (site-independent
        constants: gamma, tau, epsilon, rotations).  At least one is a field.
      - labels are local to the call.  A field's labels bind to the axes
        listed in `axes=` that it has, in the order listed there, one label
        per axis.  A raw array's labels bind to all its dimensions, in order.
        A listed axis that no field has is an error.
      - `axes=None` is allowed only when every field has at most one internal
        axis, which it then binds; a field with several needs `axes=`, so
        binding never depends on the order a field was declared in.
      - unlisted axes pass through: aligned by name, elementwise, never
        summed, just like spacetime.
      - one label binds one axis name: a label bound to differently named
        axes is an error, even when the sizes agree.  (One name may bind
        several labels: epsilon_abc phi_a dx_b dy_c.)
      - output names never depend on the order of `axes=`.  An output label
        bound to a field axis keeps that axis's name.  A label carried only by
        constants is named by what those constants touch: follow shared
        constants outward to field-bound labels; their names must be exactly
        one, of the label's size.  Otherwise it is an error pointing to
        `out_axes` (a rectangular constant builds a new space), which names
        the output labels explicitly (agreeing with any the fields fix) and
        may introduce a new space.  No name may appear twice in the result
        (matrix-valued fields are deferred).
      - the result's axes are the output labels in the order written, then
        the passed-through axes in order of first appearance.
      - boundary conditions follow the product rule over the field
        operands; raw arrays are neutral.
  * **`phi.contract(*names)`**: Einstein summation over the named internal
    axes; with no names, over all of them, giving a per-site scalar.
    Boundary conditions are unchanged.
  * **`phi.dot(chi, *names)`** and **`phi.vdot(chi, *names)`**: product, then
    contraction over the shared axes (all of them by default); unshared axes
    pass through.  `vdot` conjugates the first operand and `dot` does not, as
    `np.vdot` and `np.dot` do.  Because they default to every shared axis, a
    term written with them works for any N and any axis name.
  * **Repeated axis names** (matrix-valued fields) are designed but deferred:
    for now they stay an error at construction.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat

SITES = (4, 3)  # unequal, so a mixed-up spacetime axis shows up as a shape error
L = lat.SquareLattice(st_dims=SITES)


def _rand(key, shape, complex_=False):
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    x = jax.random.normal(k1, shape)
    if complex_:
        x = x + 1j * jax.random.normal(k2, shape)
    return x


def _field(key, axes=None, complex_=False, bc=()):
    axes = dict(axes or {})
    F = _rand(key, SITES + tuple(axes.values()), complex_)
    return lat.LatticeField(lattice=L, F=F, axes=axes, bc=bc)


def _view(field, *names):
    # field.F with its internal axes put in the order `names`.
    own = [name for name, _ in field.axes]
    assert sorted(own) == sorted(names)
    d = len(SITES)
    return jnp.transpose(
        field.F, tuple(range(d)) + tuple(d + own.index(name) for name in names)
    )


def _pauli():
    return jnp.array(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex
    )


def _levi_civita():
    eps = np.zeros((3, 3, 3))
    for perm in itertools.permutations(range(3)):
        eps[perm] = np.linalg.det(np.eye(3)[list(perm)])
    return jnp.asarray(eps)


# --- contract -----------------------------------------------------------------


def test_contract_sums_the_named_axes():
    psi = _field(0, {"spin": 4, "color": 3})

    c = psi.contract("color")
    assert isinstance(c, lat.LatticeField)
    assert c.axes == (("spin", 4),)
    assert jnp.allclose(c.F, psi.F.sum(-1))

    s = psi.contract("spin")
    assert s.axes == (("color", 3),)
    assert jnp.allclose(s.F, psi.F.sum(-2))

    both = psi.contract("spin", "color")
    assert both.axes == ()
    assert jnp.allclose(both.F, psi.F.sum((-2, -1)))


def test_contract_with_no_names_gives_a_per_site_scalar():
    psi = _field(1, {"spin": 4, "color": 3})
    c = psi.contract()
    assert c.axes == ()
    assert c.F.shape == SITES
    assert jnp.allclose(c.F, psi.F.sum((-2, -1)))


def test_contract_never_sums_over_sites():
    # Unlike numpy's sum: a field with no internal axes has nothing to contract.
    phi = _field(2)
    assert jnp.allclose(phi.contract().F, phi.F)


def test_contracting_an_axis_the_field_lacks_is_an_error():
    psi = _field(3, {"spin": 4})
    with pytest.raises(ValueError, match="color"):
        psi.contract("color")


def test_contract_keeps_the_boundary_conditions():
    psi = _field(4, {"spin": 4}, complex_=True, bc=(-1, lat.phase(1, 2)))
    c = psi.contract()
    assert c.bc == psi.bc
    for ax in range(2):
        assert jnp.allclose(c.nn_field(ax).F, psi.nn_field(ax).contract().F)


# --- dot and vdot -------------------------------------------------------------


def test_dot_contracts_every_shared_axis():
    phi, chi = _field(5, {"a": 3}), _field(6, {"a": 3})
    d = phi.dot(chi)
    assert d.axes == ()
    assert jnp.allclose(d.F, jnp.sum(phi.F * chi.F, axis=-1))


@pytest.mark.parametrize("method, np_f", [("dot", np.dot), ("vdot", np.vdot)])
def test_dot_and_vdot_follow_numpy_site_by_site(method, np_f):
    # vdot conjugates its first operand, dot does not.
    psi = _field(7, {"spin": 4}, complex_=True)
    chi = _field(8, {"spin": 4}, complex_=True)
    got = getattr(psi, method)(chi).F
    for site in [(0, 0), (1, 2), (3, 1)]:
        expected = np_f(np.asarray(psi.F[site]), np.asarray(chi.F[site]))
        assert np.isclose(got[site], expected)


def test_vdot_of_a_field_with_itself_is_its_norm():
    psi = _field(9, {"spin": 4}, complex_=True)
    n = psi.vdot(psi).F
    assert jnp.allclose(n.imag, 0.0)
    assert jnp.allclose(n.real, jnp.sum(jnp.abs(psi.F) ** 2, axis=-1))


def test_dot_over_named_axes_only():
    psi = _field(10, {"spin": 4, "color": 3})
    chi = _field(11, {"spin": 4, "color": 3})
    d = psi.dot(chi, "color")
    assert d.axes == (("spin", 4),)
    assert jnp.allclose(d.F, jnp.einsum("...sc,...sc->...s", psi.F, chi.F))


def test_dot_passes_unshared_axes_through():
    psi = _field(12, {"spin": 4, "color": 3})
    eta = _field(13, {"spin": 4})
    expected = jnp.einsum("...sc,...s->...c", psi.F, eta.F)
    for d in (psi.dot(eta), eta.dot(psi)):
        assert d.axes == (("color", 3),)
        assert jnp.allclose(d.F, expected)


def test_dot_aligns_shared_axes_by_name():
    psi = _field(14, {"spin": 4, "color": 3})
    same = lat.LatticeField(
        lattice=L, F=jnp.swapaxes(psi.F, -1, -2), axes={"color": 3, "spin": 4}
    )
    assert jnp.allclose(psi.dot(same).F, psi.dot(psi).F)


def test_dot_over_an_unshared_axis_is_an_error():
    psi = _field(15, {"spin": 4, "color": 3})
    eta = _field(16, {"spin": 4})
    with pytest.raises(ValueError, match="color"):
        psi.dot(eta, "color")


def test_a_dot_term_does_not_care_what_the_axis_is_called():
    # One O(N) density for any N and any axis name; N = 1 is the plain scalar.
    def density(p):
        return p.dot(p) + p.dot(p.nn_field(0))

    F = _rand(17, SITES + (3,))
    a = lat.LatticeField(lattice=L, F=F, axes={"a": 3})
    flavor = lat.LatticeField(lattice=L, F=F, axes={"flavor": 3})
    assert jnp.allclose(density(a).F, density(flavor).F)

    F1 = _rand(18, SITES)
    one = lat.LatticeField(lattice=L, F=F1[..., None], axes={"a": 1})
    plain = lat.LatticeField(lattice=L, F=F1)
    assert jnp.allclose(density(one).F, (plain * plain + plain * plain.nn_field(0)).F)


def test_dot_and_vdot_apply_the_product_rule():
    psi = _field(19, {"spin": 4}, complex_=True, bc=(1, lat.phase(1, 2)))
    assert psi.dot(psi).bc == (1, -1)  # i * i
    assert psi.vdot(psi).bc == (1, 1)  # conj(i) * i

    for method in ("dot", "vdot"):
        for ax in range(2):
            hop = psi.nn_field(ax)
            assert jnp.allclose(
                getattr(psi, method)(psi).nn_field(ax).F, getattr(hop, method)(hop).F
            )


# --- einsum: syntax and operands ----------------------------------------------


def test_einsum_is_numpy_einsum_site_by_site():
    phi, chi = _field(20, {"a": 3}), _field(21, {"a": 3})
    r = lat.einsum("a,a->", phi, chi)
    assert isinstance(r, lat.LatticeField)
    assert r.lattice == L
    assert r.axes == ()
    assert r.F.shape == SITES
    assert jnp.allclose(r.F, jnp.einsum("...a,...a->...", phi.F, chi.F))


def test_einsum_open_labels_become_axes():
    phi, chi = _field(22, {"a": 3}), _field(23, {"b": 2})
    r = lat.einsum("a,b->ab", phi, chi)
    assert r.axes == (("a", 3), ("b", 2))
    assert jnp.allclose(r.F, phi.F[..., :, None] * chi.F[..., None, :])


@pytest.mark.parametrize("pattern", ["...a,...a->...", "a,a"])
def test_einsum_needs_an_output_and_no_ellipsis(pattern):
    phi, chi = _field(24, {"a": 3}), _field(25, {"a": 3})
    with pytest.raises(ValueError):
        lat.einsum(pattern, phi, chi)


def test_einsum_needs_a_field():
    with pytest.raises(ValueError, match="field"):
        lat.einsum("i,i->", jnp.ones(3), jnp.ones(3))


def test_einsum_fields_share_a_lattice():
    phi = _field(26, {"a": 3})
    other = lat.LatticeField(
        lattice=lat.SquareLattice(st_dims=(4, 4)), F=jnp.ones((4, 4, 3)), axes={"a": 3}
    )
    with pytest.raises(ValueError, match="lattice"):
        lat.einsum("a,a->", phi, other)


# --- einsum: binding labels to axes -------------------------------------------


def test_einsum_a_constant_acts_on_one_named_axis():
    # What `apply(M, name)` would have been: color passes through untouched,
    # and the output label i (carried only by gamma) is named spin.
    gamma = _rand(27, (4, 4), complex_=True)
    psi = _field(28, {"spin": 4, "color": 3}, complex_=True)

    r = lat.einsum("ij,j->i", gamma, psi, axes="spin")
    assert r.axes == (("spin", 4), ("color", 3))
    assert jnp.allclose(r.F, jnp.einsum("ij,...jc->...ic", gamma, psi.F))

    # A tuple of one name is the same as the bare name.
    same = lat.einsum("ij,j->i", gamma, psi, axes=("spin",))
    assert jnp.allclose(same.F, r.F)


def test_einsum_result_axes_are_the_output_then_the_passed_through():
    h = _rand(29, (3, 3))
    psi = _field(30, {"spin": 4, "color": 3})
    r = lat.einsum("ij,j->i", h, psi, axes="color")
    assert r.axes == (("color", 3), ("spin", 4))
    assert jnp.allclose(_view(r, "spin", "color"), jnp.einsum("ij,...sj->...si", h, psi.F))


def test_einsum_passed_through_axes_align_by_name():
    psi = _field(31, {"spin": 4, "color": 3})
    chi = _field(32, {"color": 3, "spin": 4})
    r = lat.einsum("s,s->", psi, chi, axes="spin")
    assert r.axes == (("color", 3),)
    assert jnp.allclose(r.F, jnp.einsum("...sc,...cs->...c", psi.F, chi.F))


def test_einsum_binds_labels_in_the_order_of_axes():
    # axes= fixes the binding order, whatever order the field was declared in.
    psi = _field(33, {"spin": 4, "color": 3})
    expected = jnp.swapaxes(psi.F, -1, -2)
    for r in (
        lat.einsum("sc->cs", psi, axes=("spin", "color")),
        lat.einsum("cs->cs", psi, axes=("color", "spin")),
    ):
        assert r.axes == (("color", 3), ("spin", 4))
        assert jnp.allclose(r.F, expected)


def test_einsum_a_field_with_several_axes_needs_axes():
    # Binding in declared order would make the pattern depend on how the
    # field happened to be declared, so axes=None only covers fields with at
    # most one internal axis.
    psi = _field(34, {"spin": 4, "color": 3})
    with pytest.raises(ValueError, match="axes="):
        lat.einsum("sc->cs", psi)


def test_einsum_one_label_per_bound_axis():
    gamma = _rand(35, (4, 4))
    psi = _field(36, {"spin": 4, "color": 3})
    with pytest.raises(ValueError):
        lat.einsum("ij,j->i", gamma, psi, axes=("spin", "color"))  # psi binds two
    with pytest.raises(ValueError):
        lat.einsum("ij,jc->ic", gamma, psi, axes="spin")  # ... and here only one


def test_einsum_a_listed_axis_no_field_has_is_an_error():
    psi = _field(37, {"spin": 4})
    with pytest.raises(ValueError, match="taste"):
        lat.einsum("s,s->", psi, psi, axes=("spin", "taste"))


def test_one_label_cannot_bind_two_axis_names():
    # Equal sizes, different spaces: numpy would contract them; we refuse.
    phi, chi = _field(38, {"a": 3}), _field(39, {"flavor": 3})
    with pytest.raises(ValueError, match="label"):
        lat.einsum("a,a->", phi, chi)

    # Through an explicit constant, the cross-space contraction is on purpose.
    r = lat.einsum("a,ab,b->", phi, jnp.eye(3), chi)
    assert jnp.allclose(r.F, jnp.sum(phi.F * chi.F, axis=-1))


def test_one_axis_name_can_bind_several_labels():
    # The O(3) topological charge density: eps_abc phi_a dx_b dy_c, with all
    # three fields carrying the same axis "a".
    phi, dx, dy = (_field(k, {"a": 3}) for k in (40, 41, 42))
    r = lat.einsum("abc,a,b,c->", _levi_civita(), phi, dx, dy)
    assert jnp.allclose(r.F, jnp.sum(phi.F * jnp.cross(dx.F, dy.F), axis=-1))


# --- einsum: naming the output ------------------------------------------------


def test_constant_only_outputs_are_named_by_what_the_constants_touch():
    g1, g2 = _rand(43, (4, 4)), _rand(44, (4, 4))
    psi = _field(45, {"spin": 4})
    phi, chi = _field(46, {"a": 3}), _field(47, {"a": 3})
    dot = jnp.sum(phi.F * chi.F, axis=-1)[..., None]

    cases = [
        # gamma_ij psi_j: j is spin, so i is spin.
        (lat.einsum("ij,j->i", g1, psi), ("spin", 4),
         jnp.einsum("ij,...j->...i", g1, psi.F)),
        # (gamma psi)(phi . chi): phi and chi carry a, but gamma never reaches it.
        (lat.einsum("ij,j,k,k->i", g1, psi, phi, chi), ("spin", 4),
         jnp.einsum("ij,...j->...i", g1, psi.F) * dot),
        # gamma gamma psi: i reaches spin through two constants.
        (lat.einsum("ij,jk,k->i", g1, g2, psi), ("spin", 4),
         jnp.einsum("ij,jk,...k->...i", g1, g2, psi.F)),
        # the cross product eps_abc phi_a chi_b: a and b are both "a".
        (lat.einsum("abc,a,b->c", _levi_civita(), phi, chi), ("a", 3),
         jnp.cross(phi.F, chi.F)),
    ]
    for r, axis, expected in cases:
        assert r.axes == (axis,)
        assert jnp.allclose(r.F, expected)


def test_o4_with_spin_is_never_guessed():
    # a and spin both have size 4, so no size check can catch a mix-up.
    gamma, R = _rand(48, (4, 4)), _rand(49, (4, 4))
    psi = _field(50, {"spin": 4})
    phi, chi = _field(51, {"a": 4}), _field(52, {"a": 4})

    # Naming: i is gamma's index, so it is spin, however axes= is ordered.
    dot = jnp.sum(phi.F * chi.F, axis=-1)[..., None]
    expected = jnp.einsum("ij,...j->...i", gamma, psi.F) * dot
    for axes in (None, ("spin", "a"), ("a", "spin")):
        r = lat.einsum("ij,j,k,k->i", gamma, psi, phi, chi, axes=axes)
        assert r.axes == (("spin", 4),)
        assert jnp.allclose(r.F, expected)

    # Binding: gamma on spin and R on a, whichever order the field declares.
    Psi = _field(53, {"a": 4, "spin": 4})
    Psi_swapped = lat.LatticeField(
        lattice=L, F=jnp.swapaxes(Psi.F, -1, -2), axes={"spin": 4, "a": 4}
    )
    expected = jnp.einsum("ij,kl,...lj->...ik", gamma, R, Psi.F)  # (spin, a)
    for field in (Psi, Psi_swapped):
        r = lat.einsum("ij,kl,jl->ik", gamma, R, field, axes=("spin", "a"))
        assert r.axes == (("spin", 4), ("a", 4))
        assert jnp.allclose(r.F, expected)
        with pytest.raises(ValueError, match="axes="):
            lat.einsum("ij,kl,jl->ik", gamma, R, field)


def test_out_axes_names_what_the_constants_cannot():
    # (tau^a phi_a) acting on psibar: j reaches both a and flavor.
    psibar = _field(54, {"spin": 4, "flavor": 2}, complex_=True)
    phi = _field(55, {"a": 3})
    axes = ("spin", "flavor", "a")
    with pytest.raises(ValueError, match="out_axes"):
        lat.einsum("si,aij,a->sj", psibar, _pauli(), phi, axes=axes)

    r = lat.einsum("si,aij,a->sj", psibar, _pauli(), phi, axes=axes,
                   out_axes=("spin", "flavor"))
    assert r.axes == (("spin", 4), ("flavor", 2))
    assert jnp.allclose(r.F, jnp.einsum("...si,aij,...a->...sj", psibar.F, _pauli(), phi.F))

    # out_axes must agree with what the fields fix, and name every output label.
    with pytest.raises(ValueError):
        lat.einsum("si,aij,a->sj", psibar, _pauli(), phi, axes=axes,
                   out_axes=("color", "flavor"))
    with pytest.raises(ValueError):
        lat.einsum("si,aij,a->sj", psibar, _pauli(), phi, axes=axes, out_axes=("spin",))


def test_out_axes_can_introduce_a_new_space():
    # C maps a (size 3) to a new two-dimensional space.  i reaches only a,
    # but has size 2: it is not a, and only out_axes can say what it is.
    C = _rand(56, (2, 3))
    phi = _field(57, {"a": 3})
    with pytest.raises(ValueError, match="out_axes"):
        lat.einsum("ia,a->i", C, phi)
    # A constant that reaches no field axis at all needs a name too.
    with pytest.raises(ValueError, match="out_axes"):
        lat.einsum("i,a->ia", jnp.ones(2), phi)
    r = lat.einsum("ia,a->i", C, phi, out_axes=("doublet",))
    assert r.axes == (("doublet", 2),)
    assert jnp.allclose(r.F, jnp.einsum("ia,...a->...i", C, phi.F))


def test_a_name_twice_in_the_output_is_an_error_for_now():
    # i and j are both spin: a matrix-valued field, which is deferred.
    gamma = _rand(58, (4, 4))
    psi = _field(59, {"spin": 4})
    with pytest.raises(ValueError):
        lat.einsum("ij,j->ij", gamma, psi)


def test_einsum_yukawa():
    # psibar_si tau^a_ij psi_sj phi_a: spin, flavor and the O(3) index at once.
    tau = _pauli()
    psibar = _field(46, {"spin": 4, "flavor": 2}, complex_=True)
    psi = _field(47, {"spin": 4, "flavor": 2}, complex_=True)
    phi = _field(48, {"a": 3})
    axes = ("spin", "flavor", "a")

    r = lat.einsum("si,aij,sj,a->", psibar, tau, psi, phi, axes=axes)
    assert r.axes == ()
    expected = jnp.einsum("...si,aij,...sj,...a->...", psibar.F, tau, psi.F, phi.F)
    assert jnp.allclose(r.F, expected)

    # Declaring psi as (flavor, spin) changes nothing: binding follows axes=.
    psi_fs = lat.LatticeField(
        lattice=L, F=jnp.swapaxes(psi.F, -1, -2), axes={"flavor": 2, "spin": 4}
    )
    r_fs = lat.einsum("si,aij,sj,a->", psibar, tau, psi_fs, phi, axes=axes)
    assert jnp.allclose(r_fs.F, expected)


# --- einsum: boundary conditions and transformations --------------------------


def test_einsum_boundary_conditions_follow_the_product_rule():
    R = _rand(49, (3, 3))
    phi = _field(50, {"a": 3}, bc=(-1, 1))
    chi = _field(51, {"a": 3}, complex_=True, bc=(-1, lat.phase(1, 2)))

    def f(p, q):
        return lat.einsum("ab,a,b->", R, p, q)

    r = f(phi, chi)
    assert r.bc == (1, lat.phase(1, 2))
    for ax in range(2):
        for shift in (1, -1):
            hopped = f(phi.nn_field(ax, shift), chi.nn_field(ax, shift))
            assert jnp.allclose(r.nn_field(ax, shift).F, hopped.F)


def test_einsum_under_jit_and_grad():
    phi, chi = _field(52, {"a": 3}), _field(53, {"a": 3})

    jitted = jax.jit(lambda p, q: lat.einsum("a,a->", p, q))(phi, chi)
    assert jnp.allclose(jitted.F, lat.einsum("a,a->", phi, chi).F)

    g = jax.grad(lambda p: jnp.sum(p.dot(p).F))(phi)
    assert jnp.allclose(g.F, 2 * phi.F)


# --- deferred: matrix-valued fields -------------------------------------------


def test_repeated_axis_names_are_an_error_for_now():
    with pytest.raises(ValueError, match="repeated"):
        lat.LatticeField(
            lattice=L, F=jnp.ones(SITES + (3, 3)), axes=(("color", 3), ("color", 3))
        )
