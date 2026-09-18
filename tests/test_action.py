import contextlib

import jax
import jax.numpy as jnp
import pytest

import lfox.lattice as lat
from lfox.action import Action

# Action composition, per-term params, and field aliasing.  Written before the
# fix (rewrite_thoughts.md, step 1): `_S` receives a dict of fields, every term
# keeps its own immutable `params`, and `with_param` / `rebind` return new actions.


class Mass(Action):
    """S = m2/2 * phi^2 per site."""

    @staticmethod
    @jax.jit
    def _S(fields, params):
        return 0.5 * params['m2'] * fields['phi'] ** 2


class Hopping(Action):
    """S = -kappa * sum_mu phi(x) phi(x - mu) per site."""

    @staticmethod
    @jax.jit
    def _S(fields, params):
        phi = fields['phi']
        S = phi * phi.nn_field(0)
        for ax in range(1, phi.d()):
            S += phi * phi.nn_field(ax)
        return -params['kappa'] * S


# Reference values computed directly on the arrays, independent of `Action`.
def mass_S(F, m2):
    return 0.5 * m2 * jnp.sum(F**2)


def hopping_S(F, kappa):
    return -kappa * sum(jnp.sum(F * jnp.roll(F, 1, axis=ax)) for ax in range(F.ndim))


def hopping_force(F, kappa):
    return -kappa * sum(
        jnp.roll(F, 1, axis=ax) + jnp.roll(F, -1, axis=ax) for ax in range(F.ndim)
    )


def terms(action):
    return [action, *action.sub_actions]


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


# --- composition never mutates its operands ----------------------------------


def test_add_leaves_operands_unchanged():
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = Hopping(field_names=['phi'], params={'kappa': 0.1})
    c = a + b

    assert c is not a
    assert dict(a.params) == {'m2': 1.0} and a.sub_actions == []
    assert dict(b.params) == {'kappa': 0.1} and b.sub_actions == []


def test_iadd_rebinds_name_without_mutating_original():
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = Hopping(field_names=['phi'], params={'kappa': 0.1})
    a0 = a
    a += b

    assert a0.sub_actions == [] and dict(a0.params) == {'m2': 1.0}
    assert len(terms(a)) == 2


def test_nested_add_flattens_to_leaf_terms(phi):
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = Mass(field_names=['phi'], params={'m2': 3.0})
    c = Hopping(field_names=['phi'], params={'kappa': 0.1})
    bc = b + c
    abc = a + bc

    assert len(terms(abc)) == 3
    assert all(t.sub_actions == [] for t in abc.sub_actions)
    assert len(terms(bc)) == 2  # the operand is not hollowed out

    # Association does not matter.
    S = float((a + b + c).S({'phi': phi}))
    assert float(((a + b) + c).S({'phi': phi})) == pytest.approx(S)
    assert float(abc.S({'phi': phi})) == pytest.approx(S)


def test_composite_field_names_cover_every_term():
    # `Chain` and `HMC` read `action.field_names` to find every field they must
    # validate and update, so a composite must report the union of its terms.
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = Mass(field_names=['chi'], params={'m2': 1.0})
    c = Mass(field_names=['phi'], params={'m2': 3.0})

    assert sorted((a + b).field_names) == ['chi', 'phi']
    assert list((a + c).field_names) == ['phi']


# --- evaluation: dict convention and per-term params ---------------------------


def test_S_receives_fields_as_dict(phi):
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    assert float(a.S({'phi': phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_composite_S_is_sum_of_terms(phi):
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = Hopping(field_names=['phi'], params={'kappa': 0.1})
    ab = a + b

    expected = mass_S(phi.F, 1.0) + hopping_S(phi.F, 0.1)
    assert float(ab.S({'phi': phi})) == pytest.approx(float(expected))
    assert jnp.allclose(
        ab.S_field({'phi': phi}).F,
        a.S_field({'phi': phi}).F + b.S_field({'phi': phi}).F,
    )


def test_composite_dS_is_sum_of_forces(phi):
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = Hopping(field_names=['phi'], params={'kappa': 0.1})

    force = (a + b).dS({'phi': phi})['phi'].F
    assert jnp.allclose(force, 1.0 * phi.F + hopping_force(phi.F, 0.1))


def test_terms_keep_their_own_params(phi):
    # Two terms sharing a param *name* with different values: the total must
    # use each term's own value, not a merged last-write-wins dict.
    light = Mass(field_names=['phi'], params={'m2': 1.0})
    heavy = Mass(field_names=['phi'], params={'m2': 3.0})

    expected = mass_S(phi.F, 1.0 + 3.0)
    assert float((light + heavy).S({'phi': phi})) == pytest.approx(float(expected))


# --- params are immutable; with_param returns a new action ---------------------


def test_with_param_returns_new_action(phi):
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    b = a.with_param('m2', 3.0)

    assert b is not a and type(b) is type(a)
    assert a.params['m2'] == 1.0
    assert b.params['m2'] == 3.0
    # Interleave so each evaluation hits a jit cache the other has populated.
    assert float(a.S({'phi': phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))
    assert float(b.S({'phi': phi})) == pytest.approx(float(mass_S(phi.F, 3.0)))
    assert float(a.S({'phi': phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))


def test_with_param_rejects_unknown_name():
    # A typo must not silently create an unused coupling.
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    with pytest.raises(KeyError):
        a.with_param('mass', 3.0)


def test_jit_cache_survives_param_mutation(phi):
    # rewrite_thoughts.md: mutating a static params dict in place rewrote the
    # jit cache key under an already-compiled executable, so a *fresh* action
    # with the new value returned the stale result.  Whether mutation is
    # rejected or honored, every action must evaluate with the params it holds.
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    assert float(a.S({'phi': phi})) == pytest.approx(float(mass_S(phi.F, 1.0)))

    with contextlib.suppress(TypeError):
        a.params['m2'] = 3.0

    fresh = Mass(field_names=['phi'], params={'m2': 3.0})
    assert float(fresh.S({'phi': phi})) == pytest.approx(float(mass_S(phi.F, 3.0)))
    assert float(a.S({'phi': phi})) == pytest.approx(
        float(mass_S(phi.F, a.params['m2']))
    )


# --- rebind: field aliasing for many-flavor actions ----------------------------


def test_rebind_returns_new_action_on_external_name(phi):
    a = Mass(field_names=['phi'], params={'m2': 1.0})
    r = a.rebind({'phi': 'chi'})

    assert list(r.field_names) == ['chi']
    assert list(a.field_names) == ['phi']
    assert float(r.S({'chi': phi})) == pytest.approx(float(a.S({'phi': phi})))
    with pytest.raises(KeyError):
        r.S({'phi': phi})


def test_two_flavor_action_via_rebind(phi, chi):
    template = Mass(field_names=['phi'], params={'m2': 1.0})
    psi_1 = template.rebind({'phi': 'psi_1'})
    psi_2 = template.with_param('m2', 3.0).rebind({'phi': 'psi_2'})
    two_flavor = psi_1 + psi_2
    fields = {'psi_1': phi, 'psi_2': chi}

    assert sorted(two_flavor.field_names) == ['psi_1', 'psi_2']
    expected = mass_S(phi.F, 1.0) + mass_S(chi.F, 3.0)
    assert float(two_flavor.S(fields)) == pytest.approx(float(expected))

    force = two_flavor.dS(fields)
    assert set(force) == {'psi_1', 'psi_2'}
    assert jnp.allclose(force['psi_1'].F, 1.0 * phi.F)
    assert jnp.allclose(force['psi_2'].F, 3.0 * chi.F)
