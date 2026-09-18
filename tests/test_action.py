"""Design tests for `lfox.action`, written before the implementation.

These pin the revised `Action` object model:

  * **Roles from the signature.**  A term's fields are the parameter names of
    its `density` / `total` hook.  Those are the default external names;
    anything else requires an explicit `rebind`.
  * **Flat composition, one class.**  `a + b` returns a composite `Action`
    holding a flat `terms` tuple -- never `type(a)`, and never a separate
    composite type.  `terms` is uniform on leaves and composites alike, so
    `sum(...)` over a generator is the idiom for building many-flavor actions.
  * **Two verbs.**  `+` combines theories and normally introduces a field;
    if both operands have identical field sets it warns and points at
    `add_term`, the explicit path for putting another term on the same
    fields.  Both paths reject duplicates.
  * **Couplings are dataclass fields**: pytree leaves by default (no retrace
    on a coupling change, no stale-cache physics), `static=True` only for
    structural knobs.
  * **Duplicate terms are rejected.**  Two terms differing only in a traced
    coupling are an accident; distinct `label`s are how you say you meant it.
  * **Terms declare what they sample.**  A pseudofermion term owns its own
    heatbath; everything it does not sample is evolved, and `dS` follows.

Author hooks (`density`, `total`, `draw`, `gradient`) take fields as *roles*,
by name.  The public API (`S_field`, `S`, `sample`, `dS`) takes and returns
dicts keyed by *external* field name.  That is the whole of the binding layer.

Note on sign conventions: `dS` returns the gradient dS/dphi, not the force
-dS/dphi.  `gradient` overrides it with the same convention.
"""

import collections
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import lfox.lattice as lat
from lfox.action import Action, DuplicateTermError, SharedFieldsWarning

# Trace / call counters.  `density` and `gradient` bodies run only when the
# enclosing jit traces, so these count compilations, not evaluations.
TRACES = collections.Counter()
CALLS = collections.Counter()


# --- terms under test ---------------------------------------------------------


class Mass(Action):
    """S = m2/2 phi^2 per site."""

    m2: float

    def density(self, phi):
        return 0.5 * self.m2 * phi**2


class Hopping(Action):
    """S = -kappa sum_mu phi(x) phi(x-mu) per site."""

    kappa: float

    def density(self, phi):
        S = phi * phi.nn_field(0)
        for ax in range(1, phi.d()):
            S += phi * phi.nn_field(ax)
        return -self.kappa * S


class Yukawa(Action):
    """S = g phi chi^2 per site.  A two-role term, to pin binding order."""

    g: float

    def density(self, phi, chi):
        return self.g * phi * chi**2


class Polynomial(Action):
    """S = c sum_{k=1..n} phi^(2k).

    `n` is structural -- it sets the number of terms in a Python loop, so it
    cannot be traced -- while `c` is an ordinary coupling.  One class that
    exercises the static/traced split, retracing, and structural distinctness.
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


class ZeroMode(Action):
    """S = h/2 (sum_x phi)^2 / V.

    Genuinely nonlocal: there is no per-site density to write down, so this
    term supplies `total` instead.  Pins that `density` is optional.
    """

    h: float

    def total(self, phi):
        return 0.5 * self.h * jnp.sum(phi.F) ** 2 / phi.F.size


class AnalyticMass(Action):
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


class StochasticMass(Action):
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


# --- roles and binding --------------------------------------------------------


def test_roles_are_the_density_signature():
    # Writing the term forces the author to name its fields; those names are
    # the default binding, so the common case needs no binding argument at all.
    assert Mass(m2=1.0).field_names == ("phi",)
    assert Yukawa(g=1.0).field_names == ("phi", "chi")  # signature order


def test_default_binding_evaluates_under_the_declared_names(phi):
    a = Mass(m2=1.0)
    assert float(a.S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_rebind_returns_a_new_action(phi):
    a = Mass(m2=1.0)
    r = a.rebind(phi="chi")

    assert r is not a
    assert r.field_names == ("chi",) and a.field_names == ("phi",)
    assert float(r.S({"chi": phi})) == pytest.approx(float(a.S({"phi": phi})))
    with pytest.raises(KeyError):
        r.S({"phi": phi})


def test_rebind_is_partial_and_keyed_by_role():
    # Keys are always *roles*, never the current external name, so chained
    # rebinds do not require the caller to track intermediate names.
    y = Yukawa(g=1.0).rebind(chi="chi_2")
    assert y.field_names == ("phi", "chi_2")
    assert y.rebind(chi="chi_3").field_names == ("phi", "chi_3")


def test_rebind_rejects_an_unknown_role():
    # A typo must not silently create a field nothing reads.
    with pytest.raises(ValueError):
        Mass(m2=1.0).rebind(psi="psi_1")


def test_extra_fields_are_ignored(phi, chi):
    # `Chain` hands every term the full field dict; a term takes its own slice.
    a = Mass(m2=1.0)
    assert float(a.S({"phi": phi, "chi": chi})) == pytest.approx(
        float(a.S({"phi": phi}))
    )


# --- composition --------------------------------------------------------------


def test_leaf_action_is_its_own_term():
    # `terms` is uniform, so callers never special-case a single term.
    a = Mass(m2=1.0)
    assert a.terms == (a,)


def test_add_returns_a_composite_not_the_left_operand_type():
    # One class throughout: a composite is an `Action` holding `terms`, not an
    # instance of some separate composite type.
    a, b = Mass(m2=1.0), Yukawa(g=0.2)
    c = a + b

    assert type(c) is Action and type(c) is not type(a)
    assert c.terms == (a, b)
    assert (b + a).terms == (b, a)  # same type either way


def test_add_leaves_operands_unchanged():
    a, b = Mass(m2=1.0), Yukawa(g=0.2)
    a + b

    assert a.terms == (a,) and a.m2 == 1.0
    assert b.terms == (b,) and b.g == 0.2


def test_iadd_rebinds_the_name_without_mutating_the_original():
    a, b = Mass(m2=1.0), Yukawa(g=0.2)
    a0 = a
    a += b

    assert a0.terms == (a0,)
    assert len(a.terms) == 2


def test_nested_add_flattens_to_leaf_terms(phi, chi):
    a = Mass(m2=1.0)
    b = Yukawa(g=0.2)
    c = Mass(m2=3.0).rebind(phi="chi")
    bc = b + c
    abc = a + bc
    fields = {"phi": phi, "chi": chi}

    assert abc.terms == (a, b, c)  # flat, not (a, composite(b, c))
    assert bc.terms == (b, c)  # the operand is not hollowed out
    assert all(t.terms == (t,) for t in abc.terms)

    # Association does not matter.
    S = float((a + b + c).S(fields))
    assert float(((a + b) + c).S(fields)) == pytest.approx(S)
    assert float(abc.S(fields)) == pytest.approx(S)


def test_sum_builtin_composes_a_generator(phi):
    # The many-flavor idiom: `sum(term(i) for i in range(N))`.  Needs 0 + action.
    terms = [Mass(m2=1.0).rebind(phi=f"phi_{i}") for i in range(3)]
    total = sum(terms)

    assert total.terms == tuple(terms)
    fields = {f"phi_{i}": phi for i in range(3)}
    assert float(total.S(fields)) == pytest.approx(3 * float(mass_S(phi.F, 1.0)))


def test_composite_field_names_are_an_ordered_union():
    a = Mass(m2=1.0)  # phi
    b = Hopping(kappa=0.1)  # phi
    c = Yukawa(g=0.2)  # phi, chi

    assert a.add_term(b).field_names == ("phi",)
    assert (a + c).field_names == ("phi", "chi")
    assert (c + a).field_names == ("phi", "chi")


def test_composite_S_is_the_sum_of_its_terms(phi):
    a, b = Mass(m2=1.0), Hopping(kappa=0.1)
    ab = a.add_term(b)

    expected = mass_S(phi.F, 1.0) + hopping_S(phi.F, 0.1)
    assert float(ab.S({"phi": phi})) == pytest.approx(float(expected))
    assert jnp.allclose(
        ab.S_field({"phi": phi}).F,
        a.S_field({"phi": phi}).F + b.S_field({"phi": phi}).F,
    )


def test_composite_dS_is_the_sum_of_forces(phi):
    a, b = Mass(m2=1.0), Hopping(kappa=0.1)

    grad = a.add_term(b).dS({"phi": phi})["phi"].F
    assert jnp.allclose(grad, 1.0 * phi.F + hopping_force(phi.F, 0.1))


def test_terms_are_addressable_by_label():
    # Needed later so a multi-timescale integrator can ask for the force of a
    # subset of terms; here it is just the handle for inspecting a composite.
    g = Mass(m2=1.0, label="light")
    h = Mass(m2=3.0, label="heavy")
    S = g.add_term(h)

    assert S["light"] is g and S["heavy"] is h
    with pytest.raises(KeyError):
        S["charm"]


# --- two verbs: `+` vs `add_term` ---------------------------------------------


def test_add_warns_when_the_operands_have_identical_fields():
    # `+` reads as "combine two theories", which normally brings a new field
    # with it.  Identical field sets more often mean a name collision -- the
    # author meant the second term on a second field -- so `+` says so and
    # names the verb for the deliberate case.
    with pytest.warns(SharedFieldsWarning):
        Mass(m2=1.0) + Hopping(kappa=0.1)


def test_add_is_quiet_when_the_composition_introduces_a_field():
    # A gauge action plus a gauge-charged fermion: the gauge field is shared,
    # the pseudofermion is new.  The ordinary case, and it must stay silent --
    # a subset rule instead of an equality rule would wrongly fire here.
    with warnings.catch_warnings():
        warnings.simplefilter("error", SharedFieldsWarning)
        S = Hopping(kappa=0.1) + StochasticMass(g=0.5)

    assert S.field_names == ("phi", "eta")


def test_add_is_quiet_when_the_operands_share_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SharedFieldsWarning)
        Mass(m2=1.0) + Mass(m2=3.0).rebind(phi="chi")


def test_add_term_is_the_explicit_path_for_extra_terms(phi):
    a, b = Mass(m2=1.0), Hopping(kappa=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SharedFieldsWarning)
        S = a.add_term(b)

    assert S.terms == (a, b)
    expected = mass_S(phi.F, 1.0) + hopping_S(phi.F, 0.1)
    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))


def test_add_term_returns_a_new_action():
    # Actions are frozen, so a bare `S.add_term(x)` builds a new action and
    # discards it.  The API must never read as in-place; this is the same
    # mutation hazard that produced stale-cache physics in the old Action.
    a = Mass(m2=1.0)
    S = a.add_term(Hopping(kappa=0.1))

    assert S is not a
    assert a.terms == (a,)


def test_add_term_is_variadic(phi):
    # The assembly path `sum()` cannot cover, since these share every field.
    S = Mass(m2=1.0).add_term(Hopping(kappa=0.1), ZeroMode(h=0.5))
    assert len(S.terms) == 3


# --- duplicate terms ----------------------------------------------------------


def test_duplicate_term_is_rejected():
    # Two mass terms on the same field differ only in a traced coupling.  This
    # is almost always a name collision -- the author meant two fields -- and
    # never has to be written this way, since m2 = 1 + 3 is the same action.
    a = Mass(m2=1.0)
    b = Mass(m2=3.0)

    # The duplicate check runs before the shared-fields warning, so these
    # raise rather than warn.  Both verbs are checked: `add_term` is an
    # explicit "another term on these fields", not "any term at all".
    with pytest.raises(DuplicateTermError):
        a + b
    with pytest.raises(DuplicateTermError):
        a + a
    with pytest.raises(DuplicateTermError):
        a.add_term(b)
    with pytest.raises(DuplicateTermError):
        a.add_term(Hopping(kappa=0.1), b)  # caught on a flattened composite


def test_distinct_labels_declare_a_deliberate_duplicate(phi):
    # The escape hatch costs one keyword and documents the intent.
    light = Mass(m2=1.0, label="light")
    heavy = Mass(m2=3.0, label="heavy")

    S = light.add_term(heavy)
    assert float(S.S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 4.0)))


def test_different_bindings_are_not_duplicates(phi, chi):
    a = Mass(m2=1.0)
    b = Mass(m2=3.0).rebind(phi="chi")

    S = a + b  # no shared fields at all, so `+` is the right verb
    expected = mass_S(phi.F, 1.0) + mass_S(chi.F, 3.0)
    assert float(S.S({"phi": phi, "chi": chi})) == pytest.approx(float(expected))


def test_different_structure_is_not_a_duplicate(phi):
    # A term-by-term Landau-Ginzburg expansion: same class, same field,
    # different static structure.  Legitimate, must not trip the check.
    S = Polynomial(c=1.0, n=1).add_term(Polynomial(c=2.0, n=2))

    expected = jnp.sum(phi.F**2) + 2.0 * jnp.sum(phi.F**2 + phi.F**4)
    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))


def test_different_term_types_on_one_field_are_not_duplicates(phi):
    # The single most common composition in the library; the check must be
    # silent here or it is worthless.
    S = Mass(m2=1.0).add_term(Hopping(kappa=0.1), ZeroMode(h=0.5))
    assert len(S.terms) == 3


# --- per-term params ----------------------------------------------------------


def test_each_term_evaluates_with_its_own_params(phi, chi):
    # The bug this replaces: sub-actions were called with the parent's merged
    # params, so two terms sharing a coupling *name* silently shared its value.
    light = Mass(m2=1.0)
    heavy = Mass(m2=3.0).rebind(phi="chi")
    fields = {"phi": phi, "chi": chi}

    expected = mass_S(phi.F, 1.0) + mass_S(chi.F, 3.0)
    assert float((light + heavy).S(fields)) == pytest.approx(float(expected))

    grad = (light + heavy).dS(fields)
    assert jnp.allclose(grad["phi"].F, 1.0 * phi.F)
    assert jnp.allclose(grad["chi"].F, 3.0 * chi.F)


def test_params_are_immutable():
    a = Mass(m2=1.0)
    with pytest.raises((AttributeError, TypeError, eqx.errors.EquinoxRuntimeError)):
        a.m2 = 3.0


def test_with_param_returns_a_new_action(phi):
    a = Mass(m2=1.0)
    b = a.with_param("m2", 3.0)

    assert b is not a and type(b) is type(a)
    assert a.m2 == 1.0 and b.m2 == 3.0
    # Interleaved, so each evaluation hits a jit cache the other has populated.
    assert float(a.S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))
    assert float(b.S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 3.0)))
    assert float(a.S({"phi": phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_with_param_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        Mass(m2=1.0).with_param("mass", 3.0)


def test_with_param_broadcasts_over_a_composite(phi):
    # One knob, four doublets: `S.with_param('m2', mf)` retunes every term that
    # declares it and leaves the rest alone.
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


def test_with_param_on_a_composite_rejects_an_unknown_name():
    S = Mass(m2=1.0).add_term(Hopping(kappa=0.1))
    with pytest.raises(ValueError):
        S.with_param("beta", 5.4)


# --- static vs traced couplings -----------------------------------------------


def test_changing_a_coupling_does_not_retrace(phi):
    # Couplings are pytree leaves, so a coupling scan compiles once.  The value
    # assertions also rule out the old failure mode, where a second action
    # reused the first one's executable and returned its physics.
    a = Polynomial(c=1.0, n=1, tag="no-retrace")
    b = Polynomial(c=2.0, n=1, tag="no-retrace")

    before = TRACES["Polynomial"]
    Sa = float(a.S({"phi": phi}))
    Sb = float(b.S({"phi": phi}))

    assert TRACES["Polynomial"] - before == 1
    assert Sa == pytest.approx(float(jnp.sum(phi.F**2)))
    assert Sb == pytest.approx(float(2.0 * jnp.sum(phi.F**2)))


def test_changing_a_structural_field_does_retrace(phi):
    # `n` drives a Python loop, so it must be static -- and static means a new
    # jit cache entry, which is the cost that buys the loop.
    a = Polynomial(c=1.0, n=1, tag="retrace")
    b = Polynomial(c=1.0, n=2, tag="retrace")

    before = TRACES["Polynomial"]
    float(a.S({"phi": phi}))
    float(b.S({"phi": phi}))

    assert TRACES["Polynomial"] - before == 2


def test_action_is_differentiable_in_its_couplings(phi):
    # Falls out of couplings-as-leaves, and is what a parameter fit or a
    # learned action needs.  dS/dm2 = 1/2 sum phi^2.
    def S_of_m2(m2):
        return Mass(m2=m2).S({"phi": phi})

    assert float(jax.grad(S_of_m2)(1.0)) == pytest.approx(
        float(0.5 * jnp.sum(phi.F**2))
    )


# --- optional density, overridable gradient -----------------------------------


def test_a_term_may_supply_only_a_total(phi):
    z = ZeroMode(h=0.5)
    expected = 0.5 * 0.5 * jnp.sum(phi.F) ** 2 / phi.F.size

    assert float(z.S({"phi": phi})) == pytest.approx(float(expected))
    assert jnp.allclose(
        z.dS({"phi": phi})["phi"].F,
        0.5 * jnp.sum(phi.F) / phi.F.size * jnp.ones_like(phi.F),
    )
    with pytest.raises(NotImplementedError):
        z.S_field({"phi": phi})


def test_a_composite_without_a_density_everywhere_still_has_S(phi):
    S = Mass(m2=1.0).add_term(ZeroMode(h=0.5))
    expected = mass_S(phi.F, 1.0) + 0.5 * 0.5 * jnp.sum(phi.F) ** 2 / phi.F.size

    assert float(S.S({"phi": phi})) == pytest.approx(float(expected))
    with pytest.raises(NotImplementedError):
        S.S_field({"phi": phi})


def test_an_analytic_gradient_is_used_instead_of_autodiff(phi):
    a = AnalyticMass(m2=2.0)
    before = CALLS["AnalyticMass.gradient"]

    grad = a.add_term(Hopping(kappa=0.1)).dS({"phi": phi})["phi"].F

    assert CALLS["AnalyticMass.gradient"] > before
    assert jnp.allclose(grad, 2.0 * phi.F + hopping_force(phi.F, 0.1))


# --- sampled fields -----------------------------------------------------------


def test_a_term_declares_the_fields_it_samples():
    t = StochasticMass(g=0.5)

    assert t.field_names == ("phi", "eta")
    assert t.sampled_fields == ("eta",)
    assert t.evolved_fields == ("phi",)


def test_a_composite_partitions_its_fields():
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

    grad = S.dS(fields)
    assert set(grad) == {"phi"}


def test_dS_accepts_an_explicit_wrt(phi, chi):
    # The seam a Chain uses to freeze a background field, or a test uses to
    # check the heatbath gradient.
    S = Mass(m2=1.0) + StochasticMass(g=0.5)
    fields = {"phi": phi, "eta": chi}

    grad = S.dS(fields, wrt=("phi", "eta"))
    assert set(grad) == {"phi", "eta"}
    assert jnp.allclose(grad["eta"].F, chi.F / (1 + 0.5 * phi.F**2))


def test_sample_returns_external_names(phi, key):
    t = StochasticMass(g=0.5).rebind(eta="eta_2")
    drawn = t.sample({"phi": phi}, key)

    assert set(drawn) == {"eta_2"}
    assert drawn["eta_2"].F.shape == phi.F.shape


def test_sample_is_deterministic_in_its_key(phi, key):
    t = StochasticMass(g=0.5)
    assert jnp.allclose(
        t.sample({"phi": phi}, key)["eta"].F,
        t.sample({"phi": phi}, key)["eta"].F,
    )


def test_sample_sees_the_current_fields(phi, chi, key):
    # A pseudofermion heatbath draws from D_dag[U] chi: it must depend on the
    # configuration it is drawn against, not just on the key.
    t = StochasticMass(g=0.5)
    assert not jnp.allclose(
        t.sample({"phi": phi}, key)["eta"].F,
        t.sample({"phi": chi}, key)["eta"].F,
    )


def test_each_sampled_field_gets_an_independent_draw(phi, key):
    # Splitting the key once and reusing it would give every pseudofermion the
    # same noise -- a silent, catastrophic loss of stochastic independence.
    S = sum(StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(2))
    drawn = S.sample({"phi": phi}, key)

    assert set(drawn) == {"eta_0", "eta_1"}
    assert not jnp.allclose(drawn["eta_0"].F, drawn["eta_1"].F)


def test_composite_sample_returns_only_sampled_fields(phi, key):
    S = Mass(m2=1.0) + StochasticMass(g=0.5)
    assert set(S.sample({"phi": phi}, key)) == set(S.sampled_fields)


# --- the target use case ------------------------------------------------------


def test_replicate_is_the_loop_written_once(phi, chi):
    by_hand = sum(StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(4))
    by_helper = StochasticMass(g=0.5).replicate(4, "eta")

    assert by_helper.field_names == by_hand.field_names
    fields = {"phi": phi} | {f"eta_{i}": chi for i in range(4)}
    assert float(by_helper.S(fields)) == pytest.approx(float(by_hand.S(fields)))


def test_multi_flavor_assembly(phi, chi, key):
    # The acceptance test for the whole design: a gauge-like term plus four
    # degenerate two-flavor blocks, built by iteration, retuned by one knob.
    S = Hopping(kappa=0.1, label="gauge")
    S += sum(StochasticMass(g=0.5).rebind(eta=f"eta_{i}") for i in range(4))

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
