"""Design tests for `lfox.action`, written before the implementation.

These pin the revised object model, which has exactly two classes:

  * **`Term` is what you write.**  Subclass it and supply `density` (or
    `total`).  A term's fields are the parameter names of that hook -- its
    *roles* -- and those are its default external names; anything else takes
    an explicit `rebind`, keyed by role.  A term is not evaluable: it has no
    `S`, `dS` or `sample`.
  * **`Action` is what you evaluate.**  A flat, final sum of terms, built by
    `Action(*terms)`, `+`, or `sum(...)`.  It owns the public API and its own
    `rebind`, keyed by *external* field name, since roles are only unique
    within one term class.  Anything that consumes an action converts its
    input with `eqx.field(converter=Action)`, so a lone term is never seen
    downstream.
  * **One composition verb.**  `+` sums terms, whether or not they share
    fields.  No two terms may share both a `name` and their fields; distinct
    `label`s are how you say you meant a duplicate.
  * **Couplings are dataclass fields**: pytree leaves by default (no retrace
    on a coupling change, no stale-cache physics), `static=True` for anything
    structural or non-numeric.
  * **Terms declare what they sample.**  A pseudofermion term owns its own
    heatbath; everything it does not sample is evolved, and `dS` follows.

Author hooks (`density`, `total`, `draw`, `gradient`) take fields as *roles*,
by name.  The public API (`S_field`, `S`, `sample`, `dS`) takes and returns
dicts keyed by *external* field name.  That is the whole of the binding layer.

Note on sign conventions: `dS` returns the gradient dS/dphi, not the force
-dS/dphi.  `gradient` overrides it with the same convention.
"""

import collections
import math
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import lfox.lattice as lat
from lfox.action import Action, DuplicateTermError, Term

# Trace / call counters.  `density` and `gradient` bodies run only when the
# enclosing jit traces, so these count compilations, not evaluations.
TRACES = collections.Counter()
CALLS = collections.Counter()


# --- terms under test ---------------------------------------------------------


class Mass(Term):
    """S = m2/2 phi^2 per site."""

    m2: float

    def density(self, phi):
        return 0.5 * self.m2 * phi**2


class Hopping(Term):
    """S = -kappa sum_mu phi(x) phi(x-mu) per site."""

    kappa: float

    def density(self, phi):
        S = phi * phi.nn_field(0)
        for ax in range(1, phi.d()):
            S += phi * phi.nn_field(ax)
        return -self.kappa * S


class Yukawa(Term):
    """S = g phi chi^2 per site.  A two-role term, to pin binding order."""

    g: float

    def density(self, phi, chi):
        return self.g * phi * chi**2


class Polynomial(Term):
    """S = c sum_{k=1..n} phi^(2k).

    `n` is structural -- it sets the number of terms in a Python loop, so it
    cannot be traced -- while `c` is an ordinary coupling.  One class that
    exercises the static/traced split and retracing.
    """

    c: float
    n: int = eqx.field(static=True)
    tag: str = eqx.field(static=True, default="")  # test-only cache isolation

    def density(self, phi):
        TRACES["Polynomial"] += 1
        S = 0 * phi
        for k in range(1, self.n + 1):
            S = S + self.c * phi ** (2 * k)
        return S


class ZeroMode(Term):
    """S = h/2 (sum_x phi)^2 / V.

    Genuinely nonlocal: there is no per-site density to write down, so this
    term supplies `total` instead.  Pins that `density` is optional.
    """

    h: float

    def total(self, phi):
        return 0.5 * self.h * jnp.sum(phi.F) ** 2 / phi.F.size


class AnalyticMass(Term):
    """Same physics as `Mass`, but supplies its own gradient.

    Stands in for a pseudofermion force, which is written analytically rather
    than differentiated through a solver.
    """

    m2: float

    def density(self, phi):
        return 0.5 * self.m2 * phi**2

    def gradient(self, phi):
        CALLS["AnalyticMass.gradient"] += 1
        return {"phi": self.m2 * phi}


class StochasticMass(Term):
    """Toy pseudofermion: S = eta^2 / (2 (1 + g phi^2)).

    `eta` is drawn from a heatbath that depends on the current `phi`, which is
    the structure of a real pseudofermion term (phi = D_dag[U] chi) without
    needing fermions to exist yet.  Exact: eta = sqrt(1 + g phi^2) * chi with
    chi ~ N(0,1) makes S = chi^2/2.
    """

    g: float
    sampled = ("eta",)

    def density(self, phi, eta):
        return 0.5 * eta**2 / (1 + self.g * phi**2)

    def draw(self, key, phi):
        chi = jax.random.normal(key, phi.F.shape)
        return {"eta": phi.copy_new_F(chi * jnp.sqrt(1 + self.g * phi.F**2))}


class AnalyticStochasticMass(StochasticMass):
    """`StochasticMass` with a hand-written force on `phi` only.

    The shape of a real pseudofermion force: the evolved field's derivative is
    written out, the sampled field's is not.  Also pins that a concrete term
    can be subclassed.
    """

    def gradient(self, phi, eta):
        return {"phi": -self.g * phi * eta**2 / (1 + self.g * phi**2) ** 2}


# --- reference values, computed on raw arrays independently of `Action` --------


def mass_S(F, m2):
    return 0.5 * m2 * jnp.sum(F**2)


def hopping_S(F, kappa):
    return -kappa * sum(jnp.sum(F * jnp.roll(F, 1, axis=ax)) for ax in range(F.ndim))


def hopping_force(F, kappa):
    return -kappa * sum(
        jnp.roll(F, 1, axis=ax) + jnp.roll(F, -1, axis=ax) for ax in range(F.ndim)
    )


@pytest.fixture
def phi(lat4):
    return lat.LatticeField(
        lattice=lat4, F=jax.random.normal(jax.random.PRNGKey(0), lat4.st_dims)
    )


@pytest.fixture
def chi(lat4):
    return lat.LatticeField(
        lattice=lat4, F=jax.random.normal(jax.random.PRNGKey(1), lat4.st_dims)
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(12345)


# --- roles and term binding ---------------------------------------------------


def test_roles_are_the_density_signature():
    # Writing the term forces the author to name its fields; those names are
    # the default binding, so the common case needs no binding argument at all.
    assert Mass(m2=1.0).field_names == ("phi",)
    assert Yukawa(g=1.0).field_names == ("phi", "chi")  # signature order


def test_default_binding_evaluates_under_the_declared_names(phi):
    S = Action(Mass(m2=1.0))
    assert float(S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_term_rebind_returns_a_new_term(phi):
    t = Mass(m2=1.0)
    r = t.rebind(phi="chi")

    assert r is not t and type(r) is Mass
    assert r.field_names == ("chi",) and t.field_names == ("phi",)
    assert float(Action(r).S({"chi": phi})) == pytest.approx(
        float(Action(t).S({"phi": phi}))
    )
    with pytest.raises(KeyError):
        Action(r).S({"phi": phi})


def test_term_rebind_is_partial_and_keyed_by_role():
    # On a term, keys are always *roles*, never the current external name, so
    # chained rebinds do not require the caller to track intermediate names.
    y = Yukawa(g=1.0).rebind(chi="chi_2")
    assert y.field_names == ("phi", "chi_2")
    assert y.rebind(chi="chi_3").field_names == ("phi", "chi_3")


def test_term_rebind_rejects_an_unknown_role():
    # A typo must not silently create a field nothing reads.
    with pytest.raises(ValueError):
        Mass(m2=1.0).rebind(psi="psi_1")


def test_binding_must_name_every_role_once():
    # `binding` is what `rebind` writes; set by hand, it must still line up
    # with the roles or the term would read the wrong fields.
    with pytest.raises(ValueError):
        Mass(m2=1.0, binding=("a", "b"))
    with pytest.raises(ValueError):
        Yukawa(g=1.0, binding=("a",))


def test_extra_fields_are_ignored(phi, chi):
    # `Chain` hands every term the full field dict; a term takes its own slice.
    S = Action(Mass(m2=1.0))
    assert float(S.S({"phi": phi, "chi": chi})) == pytest.approx(
        float(S.S({"phi": phi}))
    )


# --- terms are written, actions are evaluated ---------------------------------


def test_a_term_is_not_evaluable():
    # A lone term and a one-term action would otherwise disagree on `rebind`
    # (role vs external name), so only `Action` carries the public API.
    t = Mass(m2=1.0)
    for name in ("S", "S_field", "dS", "sample", "terms",
                 "sampled_fields", "evolved_fields", "__call__"):
        assert not hasattr(t, name), name


def test_calling_an_action_evaluates_it(phi):
    # S(fields) reads like the physics, S[phi]; `S.S` stays as the long form.
    S = Mass(m2=1.0) + Hopping(kappa=0.1)
    assert float(S({"phi": phi})) == pytest.approx(float(S.S({"phi": phi})))


def test_action_wraps_a_single_term(phi):
    t = Mass(m2=1.0)
    S = Action(t)

    assert type(S) is Action
    assert S.terms == (t,)
    assert float(S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_action_constructor_is_variadic_and_flattens():
    a, b, c = Mass(m2=1.0), Hopping(kappa=0.1), ZeroMode(h=0.5)

    assert Action(a, b, c).terms == (a, b, c)
    assert Action(a, Action(b, c)).terms == (a, b, c)
    assert Action(Action(a, b, c)).terms == (a, b, c)


def test_action_rejects_an_empty_sum_and_non_terms():
    with pytest.raises(ValueError):
        Action()
    with pytest.raises(TypeError):
        Action(Mass(m2=1.0), 3.0)


def test_a_consumer_converts_a_term_to_an_action():
    # The pattern an evolver uses: one line, and a bare term never gets past
    # construction.  Needs `Action(term)` and `Action(action)` both to work.
    class Consumer(eqx.Module):
        action: Action = eqx.field(converter=Action)

    t = Mass(m2=1.0)
    assert type(Consumer(action=t).action) is Action
    assert Consumer(action=t).action.terms == (t,)

    S = t + Hopping(kappa=0.1)
    assert Consumer(action=S).action.terms == S.terms


# --- composition --------------------------------------------------------------


def test_add_returns_an_action():
    a, b = Mass(m2=1.0), Yukawa(g=0.2)

    assert type(a + b) is Action
    assert (a + b).terms == (a, b)
    assert (b + a).terms == (b, a)


def test_add_on_shared_fields_is_quiet(phi):
    # Kinetic plus potential on one field is the most common composition
    # there is; `+` must not warn about it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S = Mass(m2=1.0) + Hopping(kappa=0.1)

    expected = mass_S(phi.F, 1.0) + hopping_S(phi.F, 0.1)
    assert float(S({"phi": phi})) == pytest.approx(float(expected))


def test_add_leaves_operands_unchanged():
    a, b = Mass(m2=1.0), Yukawa(g=0.2)
    ab = a + b
    ab + Hopping(kappa=0.1)

    assert a.m2 == 1.0 and b.g == 0.2
    assert ab.terms == (a, b)


def test_iadd_rebinds_the_name_without_mutating_the_original():
    a, b = Mass(m2=1.0), Yukawa(g=0.2)
    S = a + b
    S0 = S
    S += Hopping(kappa=0.1)

    assert S0.terms == (a, b)
    assert len(S.terms) == 3


def test_nested_add_flattens_to_terms(phi, chi):
    a = Mass(m2=1.0)
    b = Yukawa(g=0.2)
    c = Mass(m2=3.0).rebind(phi="chi")
    bc = b + c
    fields = {"phi": phi, "chi": chi}

    assert (a + bc).terms == (a, b, c)  # term + action
    assert ((a + b) + c).terms == (a, b, c)  # action + term
    assert ((a + b) + Action(c)).terms == (a, b, c)  # action + action
    assert bc.terms == (b, c)  # the operand is not hollowed out
    assert all(isinstance(t, Term) for t in (a + bc).terms)

    S_num = float((a + b + c).S(fields))
    assert float((a + bc).S(fields)) == pytest.approx(S_num)


def test_sum_builtin_composes_a_generator(phi):
    # The many-flavor idiom: `sum(term(i) for i in range(N))`.  Needs 0 + term.
    terms = [Mass(m2=1.0).rebind(phi=f"phi_{i}") for i in range(3)]
    total = sum(terms)

    assert type(total) is Action
    assert total.terms == tuple(terms)
    fields = {f"phi_{i}": phi for i in range(3)}
    assert float(total.S(fields)) == pytest.approx(3 * float(mass_S(phi.F, 1.0)))


def test_sum_of_one_term_is_an_action():
    # A sum is an Action however many terms go in.
    assert type(sum([Mass(m2=1.0)])) is Action


def test_action_field_names_are_an_ordered_union():
    a = Mass(m2=1.0)  # phi
    b = Hopping(kappa=0.1)  # phi
    c = Yukawa(g=0.2)  # phi, chi

    assert (a + b).field_names == ("phi",)
    assert (a + c).field_names == ("phi", "chi")
    assert (c + a).field_names == ("phi", "chi")


def test_action_S_is_the_sum_of_its_terms(phi):
    a, b = Mass(m2=1.0), Hopping(kappa=0.1)
    S = a + b

    expected = mass_S(phi.F, 1.0) + hopping_S(phi.F, 0.1)
    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))
    assert jnp.allclose(
        S.S_field({"phi": phi}).F,
        Action(a).S_field({"phi": phi}).F + Action(b).S_field({"phi": phi}).F,
    )


def test_action_dS_is_the_sum_of_gradients(phi):
    grad = (Mass(m2=1.0) + Hopping(kappa=0.1)).dS({"phi": phi})["phi"].F
    assert jnp.allclose(grad, 1.0 * phi.F + hopping_force(phi.F, 0.1))


def test_sum_of_same_field_terms(phi):
    S = sum([Mass(m2=1.0), Hopping(kappa=0.1), ZeroMode(h=0.5)])

    assert len(S.terms) == 3
    expected = (
        mass_S(phi.F, 1.0)
        + hopping_S(phi.F, 0.1)
        + 0.5 * 0.5 * jnp.sum(phi.F) ** 2 / phi.F.size
    )
    assert float(S({"phi": phi})) == pytest.approx(float(expected))


# --- lookup by name -----------------------------------------------------------


def test_terms_are_addressable_by_name():
    # Needed later so a multi-timescale integrator can ask for the force of a
    # subset of terms; here it is just the handle for inspecting an action.
    # The name is the label if given, the class name otherwise.
    g = Mass(m2=1.0, label="light")
    h = Mass(m2=3.0, label="heavy")
    k = Hopping(kappa=0.1)
    S = g + h + k

    assert S["light"] is g and S["heavy"] is h and S["Hopping"] is k
    with pytest.raises(KeyError):
        S["charm"]


def test_an_ambiguous_name_raises():
    # Two unlabelled Mass terms on different fields are legal, but `S["Mass"]`
    # cannot know which one you mean; returning the first would hand an
    # integrator the wrong force.
    S = Mass(m2=1.0) + Mass(m2=3.0).rebind(phi="chi")
    with pytest.raises(KeyError):
        S["Mass"]


# --- duplicate terms ----------------------------------------------------------


def test_duplicate_term_is_rejected():
    # Two mass terms on the same field.  This is almost always a name
    # collision -- the author meant two fields -- and never has to be written
    # this way, since m2 = 1 + 3 is the same action.
    a = Mass(m2=1.0)
    b = Mass(m2=3.0)

    with pytest.raises(DuplicateTermError):
        a + b
    with pytest.raises(DuplicateTermError):
        a + a
    with pytest.raises(DuplicateTermError):
        Action(a, b)
    with pytest.raises(DuplicateTermError):
        sum([a, Hopping(kappa=0.1), b])  # caught on a flattened action


def test_a_forgotten_rebind_in_a_flavor_loop_is_rejected():
    # The failure this check exists for.  Without it, four identical terms on
    # `eta` would give 4x one flavor's action, and `sample` would draw `eta`
    # four times over -- wrong physics, no error.
    with pytest.raises(DuplicateTermError):
        sum(StochasticMass(g=0.5) for _ in range(4))


def test_distinct_labels_declare_a_deliberate_duplicate(phi):
    # The escape hatch costs one keyword and documents the intent.
    light = Mass(m2=1.0, label="light")
    heavy = Mass(m2=3.0, label="heavy")

    S = light + heavy
    assert float(S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 4.0)))


def test_one_label_on_the_same_fields_is_a_duplicate():
    # A label is an address; two terms answering to it on the same fields
    # would make `S[label]` ambiguous.  One rule covers both cases.
    with pytest.raises(DuplicateTermError):
        Mass(m2=1.0, label="x") + Hopping(kappa=0.1, label="x")


def test_different_bindings_are_not_duplicates(phi, chi):
    S = Mass(m2=1.0) + Mass(m2=3.0).rebind(phi="chi")

    expected = mass_S(phi.F, 1.0) + mass_S(chi.F, 3.0)
    assert float(S({"phi": phi, "chi": chi})) == pytest.approx(float(expected))


def test_structure_alone_does_not_distinguish_terms(phi):
    # Same class, same field, different static structure: a term-by-term
    # Landau-Ginzburg expansion.  Legitimate, but the rule is (name, fields)
    # and nothing subtler, so it takes labels to say so.
    with pytest.raises(DuplicateTermError):
        Polynomial(c=1.0, n=1) + Polynomial(c=2.0, n=2)

    S = Polynomial(c=1.0, n=1, label="quadratic") + Polynomial(
        c=2.0, n=2, label="quartic"
    )
    expected = jnp.sum(phi.F**2) + 2.0 * jnp.sum(phi.F**2 + phi.F**4)
    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))


def test_different_term_types_on_one_field_are_not_duplicates():
    # The single most common composition in the library; the check must be
    # silent here or it is worthless.
    S = Mass(m2=1.0) + Hopping(kappa=0.1) + ZeroMode(h=0.5)
    assert len(S.terms) == 3


# --- rebind on an action ------------------------------------------------------


def test_action_rebind_is_keyed_by_field_name(phi, chi):
    # Roles are only unique within one class: here both terms have a role
    # `phi`, bound to different fields.  On an action the key is the external
    # name, so only the term actually on `phi` moves.
    S = Mass(m2=1.0) + Mass(m2=3.0).rebind(phi="chi")
    R = S.rebind(phi="psi")

    assert R.field_names == ("psi", "chi")
    assert S.field_names == ("phi", "chi")  # the original is untouched
    expected = mass_S(phi.F, 1.0) + mass_S(chi.F, 3.0)
    assert float(R.S({"psi": phi, "chi": chi})) == pytest.approx(float(expected))


def test_action_rebind_reaches_every_term_on_the_field():
    # The converse: one field under different roles in different terms.
    # Here `phi` is Mass's role `phi` and Yukawa's role `chi`.
    S = Mass(m2=1.0) + Yukawa(g=0.2).rebind(phi="sigma", chi="phi")
    R = S.rebind(phi="psi")

    assert [t.field_names for t in R.terms] == [("psi",), ("sigma", "psi")]
    assert R.field_names == ("psi", "sigma")


def test_action_rebind_rejects_an_unknown_field():
    S = Mass(m2=1.0) + Hopping(kappa=0.1)
    with pytest.raises(ValueError):
        S.rebind(chi="psi")


def test_action_rebind_may_permute_but_not_merge_fields():
    # Renaming onto a name that is already in use would silently merge two
    # fields; a swap keeps them distinct and is fine.
    S = Mass(m2=1.0) + Mass(m2=3.0).rebind(phi="chi")

    swapped = S.rebind(phi="chi", chi="phi")
    assert [t.m2 for t in swapped.terms] == [1.0, 3.0]
    assert [t.field_names for t in swapped.terms] == [("chi",), ("phi",)]

    with pytest.raises(ValueError):
        S.rebind(chi="phi")


def test_term_replicate_is_the_loop_written_once(phi, chi):
    by_hand = sum(StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(4))
    by_helper = StochasticMass(g=0.5).replicate(4, "eta")

    assert type(by_helper) is Action
    assert by_helper.field_names == by_hand.field_names
    fields = {"phi": phi} | {f"eta_{i}": chi for i in range(4)}
    assert float(by_helper.S(fields)) == pytest.approx(float(by_hand.S(fields)))


def test_action_replicate_copies_a_multi_term_block(phi, chi):
    # A flavor block of two terms on one pseudofermion (the shape of a
    # Hasenbusch-split determinant), replicated by external field name.
    block = StochasticMass(g=0.5) + Mass(m2=1.0, label="eta_mass").rebind(phi="eta")
    S = block.replicate(2, "eta")

    assert S.field_names == ("phi", "eta_0", "eta_1")
    assert len(S.terms) == 4
    fields = {"phi": phi, "eta_0": chi, "eta_1": chi}
    one = block.S({"phi": phi, "eta": chi})
    assert float(S(fields)) == pytest.approx(2 * float(one))


# --- couplings ----------------------------------------------------------------


def test_each_term_evaluates_with_its_own_couplings(phi, chi):
    # The bug this replaces: sub-actions were called with the parent's merged
    # params, so two terms sharing a coupling *name* silently shared its value.
    S = Mass(m2=1.0) + Mass(m2=3.0).rebind(phi="chi")
    fields = {"phi": phi, "chi": chi}

    expected = mass_S(phi.F, 1.0) + mass_S(chi.F, 3.0)
    assert float(S.S(fields)) == pytest.approx(float(expected))

    grad = S.dS(fields)
    assert jnp.allclose(grad["phi"].F, 1.0 * phi.F)
    assert jnp.allclose(grad["chi"].F, 3.0 * chi.F)


def test_couplings_are_immutable():
    t = Mass(m2=1.0)
    with pytest.raises(AttributeError):
        t.m2 = 3.0


def test_term_with_param_returns_a_new_term(phi):
    a = Mass(m2=1.0)
    b = a.with_param("m2", 3.0)

    assert b is not a and type(b) is type(a)
    assert a.m2 == 1.0 and b.m2 == 3.0
    # Interleaved, so each evaluation hits a jit cache the other has populated.
    assert float(Action(a).S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))
    assert float(Action(b).S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 3.0)))
    assert float(Action(a).S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_term_with_param_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        Mass(m2=1.0).with_param("mass", 3.0)


def test_with_param_broadcasts_over_terms_of_one_class(phi):
    # One knob, four doublets: `S.with_param('m2', mf)` retunes every term
    # that declares it, provided they are all the same kind of term.
    S = Hopping(kappa=0.1) + sum(
        Mass(m2=1.0).rebind(phi=f"phi_{i}") for i in range(4)
    )
    retuned = S.with_param("m2", 0.25)

    fields = {"phi": phi} | {f"phi_{i}": phi for i in range(4)}
    expected = hopping_S(phi.F, 0.1) + 4 * mass_S(phi.F, 0.25)
    assert float(retuned.S(fields)) == pytest.approx(float(expected))
    assert retuned["Hopping"].kappa == 0.1
    # The original is untouched.
    assert float(S.S(fields)) == pytest.approx(
        float(hopping_S(phi.F, 0.1) + 4 * mass_S(phi.F, 1.0))
    )


def test_with_param_refuses_to_broadcast_across_classes():
    # Two different kinds of term that happen to share a coupling name are
    # not one knob; retuning both is the shared-name bug in a new place.
    S = Mass(m2=1.0) + AnalyticMass(m2=2.0).rebind(phi="chi")
    with pytest.raises(ValueError):
        S.with_param("m2", 0.5)


def test_with_param_can_target_one_term_by_name():
    S = Mass(m2=1.0) + AnalyticMass(m2=2.0).rebind(phi="chi")
    retuned = S.with_param("m2", 0.5, term="AnalyticMass")

    assert retuned["Mass"].m2 == 1.0
    assert retuned["AnalyticMass"].m2 == 0.5
    with pytest.raises(ValueError):
        S.with_param("kappa", 0.5, term="Mass")  # Mass has no kappa
    with pytest.raises(KeyError):
        S.with_param("m2", 0.5, term="charm")


def test_action_with_param_rejects_an_unknown_name():
    S = Mass(m2=1.0) + Hopping(kappa=0.1)
    with pytest.raises(ValueError):
        S.with_param("beta", 5.4)


# --- static vs traced couplings -----------------------------------------------


def test_changing_a_coupling_does_not_retrace(phi):
    # Couplings are pytree leaves, so a coupling scan compiles once.  The value
    # assertions also rule out the old failure mode, where a second action
    # reused the first one's executable and returned its physics.
    a = Action(Polynomial(c=1.0, n=1, tag="no-retrace"))
    b = Action(Polynomial(c=2.0, n=1, tag="no-retrace"))

    before = TRACES["Polynomial"]
    Sa = float(a.S({"phi": phi}))
    Sb = float(b.S({"phi": phi}))

    assert TRACES["Polynomial"] - before == 1
    assert Sa == pytest.approx(float(jnp.sum(phi.F**2)))
    assert Sb == pytest.approx(float(2.0 * jnp.sum(phi.F**2)))


def test_changing_a_structural_field_does_retrace(phi):
    # `n` drives a Python loop, so it must be static -- and static means a new
    # jit cache entry, which is the cost that buys the loop.
    a = Action(Polynomial(c=1.0, n=1, tag="retrace"))
    b = Action(Polynomial(c=1.0, n=2, tag="retrace"))

    before = TRACES["Polynomial"]
    float(a.S({"phi": phi}))
    float(b.S({"phi": phi}))

    assert TRACES["Polynomial"] - before == 2


def test_action_is_differentiable_in_its_couplings(phi):
    # Falls out of couplings-as-leaves, and is what a parameter fit or a
    # learned action needs.  dS/dm2 = 1/2 sum phi^2.
    def S_of_m2(m2):
        return Action(Mass(m2=m2)).S({"phi": phi})

    assert float(jax.grad(S_of_m2)(1.0)) == pytest.approx(
        float(0.5 * jnp.sum(phi.F**2))
    )


def test_a_non_numeric_coupling_must_be_static():
    # A non-static field is a traced pytree leaf, and a string cannot be
    # traced.  Say so at construction, not deep inside jit.
    class Shaped(Term):
        c: float
        shape: str = "quadratic"

        def density(self, phi):
            return self.c * phi**2

    with pytest.raises(TypeError, match="static"):
        Shaped(c=1.0)


def test_a_module_valued_coupling_is_allowed(phi):
    # A coupling may itself be a pytree of numbers -- a Dirac operator held by
    # a pseudofermion term, say.  Only its leaves have to be numeric.
    class Scale(eqx.Module):
        s: float

    class ScaledMass(Term):
        scale: Scale

        def density(self, phi):
            return 0.5 * self.scale.s * phi**2

    S = Action(ScaledMass(scale=Scale(s=2.0)))
    assert float(S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 2.0)))


# --- what a term class must declare -------------------------------------------


def test_an_intermediate_base_may_omit_the_density(phi):
    # Shared helpers without physics of their own: a family of gauge actions
    # sharing loop code, or of fermion terms sharing a pseudofermion heatbath.
    class Quadratic(Term):
        def square(self, phi):
            return phi**2

    class ScaledSquare(Quadratic):
        a: float

        def density(self, phi):
            return self.a * self.square(phi)

    S = Action(ScaledSquare(a=2.0))
    assert float(S.S({"phi": phi})) == pytest.approx(float(2.0 * jnp.sum(phi.F**2)))


def test_instantiating_a_term_without_density_or_total_raises():
    class Quadratic(Term):
        def square(self, phi):
            return phi**2

    with pytest.raises(TypeError):
        Quadratic()


def test_a_variadic_hook_is_rejected():
    with pytest.raises(TypeError):

        class Bad(Term):
            def density(self, *fields):
                return fields[0] ** 2


def test_a_hook_naming_an_unknown_role_is_rejected():
    with pytest.raises(TypeError):

        class Bad(Term):
            def density(self, phi):
                return phi**2

            def gradient(self, psi):
                return {"psi": 2 * psi}


def test_sampling_an_unknown_role_is_rejected():
    with pytest.raises(TypeError):

        class Bad(Term):
            sampled = ("psi",)

            def density(self, phi):
                return phi**2

            def draw(self, key):
                return {}


def test_sampling_without_a_draw_is_rejected():
    # Otherwise this would surface only when a chain first calls `sample`.
    with pytest.raises(TypeError):

        class Bad(Term):
            sampled = ("eta",)

            def density(self, phi, eta):
                return eta**2


def test_a_draw_returning_an_undeclared_role_raises(phi, key):
    class Leaky(Term):
        sampled = ("eta",)

        def density(self, phi, eta):
            return eta**2

        def draw(self, key, phi):
            return {"eta": phi, "phi": phi}

    with pytest.raises(ValueError):
        Action(Leaky()).sample({"phi": phi}, key)


# --- optional density, overridable gradient -----------------------------------


def test_a_term_may_supply_only_a_total(phi):
    S = Action(ZeroMode(h=0.5))
    expected = 0.5 * 0.5 * jnp.sum(phi.F) ** 2 / phi.F.size

    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))
    assert jnp.allclose(
        S.dS({"phi": phi})["phi"].F,
        0.5 * jnp.sum(phi.F) / phi.F.size * jnp.ones_like(phi.F),
    )
    with pytest.raises(NotImplementedError):
        S.S_field({"phi": phi})


def test_an_action_without_a_density_everywhere_still_has_S(phi):
    S = Mass(m2=1.0) + ZeroMode(h=0.5)
    expected = mass_S(phi.F, 1.0) + 0.5 * 0.5 * jnp.sum(phi.F) ** 2 / phi.F.size

    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))
    with pytest.raises(NotImplementedError):
        S.S_field({"phi": phi})


def test_an_analytic_gradient_is_used_instead_of_autodiff(phi):
    before = CALLS["AnalyticMass.gradient"]

    grad = (AnalyticMass(m2=2.0) + Hopping(kappa=0.1)).dS({"phi": phi})["phi"].F

    assert CALLS["AnalyticMass.gradient"] > before
    assert jnp.allclose(grad, 2.0 * phi.F + hopping_force(phi.F, 0.1))


def test_an_analytic_gradient_agrees_with_autodiff(phi, chi):
    fields = {"phi": phi, "eta": chi}
    analytic = Action(AnalyticStochasticMass(g=0.5)).dS(fields)["phi"].F
    autodiff = Action(StochasticMass(g=0.5)).dS(fields)["phi"].F

    assert jnp.allclose(analytic, autodiff)


def test_an_analytic_gradient_must_cover_every_requested_field(phi, chi):
    # A term with a hand-written gradient is never differentiated, so a field
    # it omits would silently get zero -- exactly when a test asks for the
    # sampled field's gradient to check a heatbath.
    S = Action(AnalyticStochasticMass(g=0.5))
    fields = {"phi": phi, "eta": chi}

    with pytest.raises(ValueError):
        S.dS(fields, wrt=("phi", "eta"))


# --- sampled fields -----------------------------------------------------------


def test_a_term_declares_the_fields_it_samples():
    S = Action(StochasticMass(g=0.5))

    assert S.field_names == ("phi", "eta")
    assert S.sampled_fields == ("eta",)
    assert S.evolved_fields == ("phi",)


def test_an_action_partitions_its_fields():
    S = Mass(m2=1.0) + sum(
        StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(2)
    )

    assert S.field_names == ("phi", "eta_0", "eta_1")
    assert S.sampled_fields == ("eta_0", "eta_1")
    assert S.evolved_fields == ("phi",)


def test_sampled_wins_over_evolved_across_terms():
    # A field sampled by any term is sampled, even if another term merely reads
    # it.  Otherwise it would get both a heatbath and a momentum.
    S = StochasticMass(g=0.5) + Mass(m2=1.0, label="eta_mass").rebind(phi="eta")

    assert S.sampled_fields == ("eta",)
    assert S.evolved_fields == ("phi",)


def test_dS_defaults_to_the_evolved_fields(phi, chi):
    S = Mass(m2=1.0) + StochasticMass(g=0.5)
    fields = {"phi": phi, "eta": chi}

    assert set(S.dS(fields)) == {"phi"}


def test_dS_accepts_an_explicit_wrt(phi, chi):
    # The seam a Chain uses to freeze a background field, or a test uses to
    # check the heatbath gradient.
    S = Mass(m2=1.0) + StochasticMass(g=0.5)
    fields = {"phi": phi, "eta": chi}

    grad = S.dS(fields, wrt=("phi", "eta"))
    assert set(grad) == {"phi", "eta"}
    assert jnp.allclose(grad["eta"].F, chi.F / (1 + 0.5 * phi.F**2))


def test_sample_returns_external_names(phi, key):
    S = Action(StochasticMass(g=0.5).rebind(eta="eta_2"))
    drawn = S.sample({"phi": phi}, key)

    assert set(drawn) == {"eta_2"}
    assert drawn["eta_2"].F.shape == phi.F.shape


def test_sample_is_deterministic_in_its_key(phi, key):
    S = Action(StochasticMass(g=0.5))
    assert jnp.allclose(
        S.sample({"phi": phi}, key)["eta"].F,
        S.sample({"phi": phi}, key)["eta"].F,
    )


def test_sample_sees_the_current_fields(phi, chi, key):
    # A pseudofermion heatbath draws from D_dag[U] chi: it must depend on the
    # configuration it is drawn against, not just on the key.
    S = Action(StochasticMass(g=0.5))
    assert not jnp.allclose(
        S.sample({"phi": phi}, key)["eta"].F,
        S.sample({"phi": chi}, key)["eta"].F,
    )


def test_each_sampled_field_gets_an_independent_draw(phi, key):
    # Splitting the key once and reusing it would give every pseudofermion the
    # same noise -- a silent, catastrophic loss of stochastic independence.
    S = sum(StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(2))
    drawn = S.sample({"phi": phi}, key)

    assert set(drawn) == {"eta_0", "eta_1"}
    assert not jnp.allclose(drawn["eta_0"].F, drawn["eta_1"].F)


def test_sample_does_not_depend_on_how_the_sum_was_grouped(phi, key):
    # Keys are split once per term of the flat sum.  A nested sum splitting
    # per child would draw different noise for the same action.
    a, b, c = (StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(3))
    left = ((a + b) + c).sample({"phi": phi}, key)
    right = (a + (b + c)).sample({"phi": phi}, key)

    assert all(jnp.array_equal(left[k].F, right[k].F) for k in left)


def test_action_sample_returns_only_sampled_fields(phi, key):
    S = Mass(m2=1.0) + StochasticMass(g=0.5)
    assert set(S.sample({"phi": phi}, key)) == set(S.sampled_fields)


def test_heatbath_draws_from_the_exact_conditional(phi):
    # A heatbath drawing eta from exp(-S[phi, eta]) at fixed phi, with S
    # quadratic in eta, gives <S> = V/2 by equipartition.  Dropping the
    # sqrt(1 + g phi^2) factor from `draw` gives ~0.39 V here, a 29-sigma miss.
    S = Action(StochasticMass(g=0.5))
    n, V = 500, phi.F.size
    keys = jax.random.split(jax.random.PRNGKey(7), n)

    draws = jax.vmap(lambda k: S.S({"phi": phi} | S.sample({"phi": phi}, k)))(keys)

    sigma = math.sqrt(0.5 / (V * n))  # std. error of mean(S)/V
    assert float(jnp.mean(draws)) / V == pytest.approx(0.5, abs=5 * sigma)


# --- inside an evolver --------------------------------------------------------


def test_the_public_api_works_under_an_outer_jit(phi, chi, key):
    # An evolver holds the action as a pytree argument and calls it inside
    # its own jit, so the couplings arrive as tracers.
    S = Mass(m2=1.0) + StochasticMass(g=0.5)
    fields = {"phi": phi, "eta": chi}

    @jax.jit
    def step(action, fields, key):
        return action.S(fields), action.dS(fields), action.sample(fields, key)

    s, grad, drawn = step(S, fields, key)

    assert float(s) == pytest.approx(float(S.S(fields)))
    assert jnp.allclose(grad["phi"].F, S.dS(fields)["phi"].F)
    assert jnp.allclose(drawn["eta"].F, S.sample(fields, key)["eta"].F)


# --- the target use case ------------------------------------------------------


def test_multi_flavor_assembly(phi, key):
    # The acceptance test for the whole design: a gauge-like term plus four
    # degenerate two-flavor blocks, built by iteration, retuned by one knob.
    S = Hopping(kappa=0.1, label="gauge")
    S += sum(StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(4))

    assert type(S) is Action
    assert S.field_names == ("phi", "eta_0", "eta_1", "eta_2", "eta_3")
    assert S.evolved_fields == ("phi",)
    assert S.sampled_fields == ("eta_0", "eta_1", "eta_2", "eta_3")

    fields = {"phi": phi} | S.sample({"phi": phi}, key)
    assert set(fields) == set(S.field_names)

    # One knob sets the mass of all four doublets at once.
    heavier = S.with_param("g", 2.0)
    assert all(t.g == 2.0 for t in heavier.terms if isinstance(t, StochasticMass))
    assert heavier["gauge"].kappa == 0.1
    assert float(heavier.S(fields)) != pytest.approx(float(S.S(fields)))

    # Only the evolved field carries a force.
    assert set(S.dS(fields)) == {"phi"}
