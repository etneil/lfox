"""Tests for `lfox.scalar.observables`, written before the implementation.

Two layers, as in the module:

  * Per-configuration observables are pinned deterministically: against the
    written-out definition in NumPy, and against exact identities between
    them (sum_t C(t) = G(0) = V m^2) and with the action.
  * Ensemble estimators are pinned against the free field, where everything
    is closed-form (`tests.free_field`): once on exact expectation values,
    deterministically, and once end to end on exact samples, statistically.

Lattices are deliberately non-cubic, so that an axis mix-up cannot hide.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action
from lfox.scalar import (
    binder,
    correlator,
    effective_mass,
    magnetization,
    nn_product,
    structure_factor,
    susceptibility,
    xi_2nd,
)
from lfox.stats import jackknife
from tests import free_field
from tests.phi4 import ScalarTerm

DIMS = (6, 4, 8)


@pytest.fixture
def lattice():
    return lat.SquareLattice(st_dims=DIMS)


@pytest.fixture
def phi(lattice):
    F = jax.random.normal(jax.random.PRNGKey(0), DIMS)
    return lat.LatticeField(lattice=lattice, F=F)


def const(lattice, c):
    return lat.LatticeField(lattice=lattice, F=jnp.full(DIMS, c))


# --- per configuration --------------------------------------------------------


def test_magnetization(lattice, phi):
    assert magnetization(const(lattice, 1.5)) == pytest.approx(1.5)
    assert magnetization(phi) == pytest.approx(np.mean(np.asarray(phi.F)))


def test_nn_product_definition(lattice, phi):
    F = np.asarray(phi.F)
    expected = np.mean([np.mean(F * np.roll(F, -1, axis=ax)) for ax in range(3)])

    assert nn_product(const(lattice, 1.5)) == pytest.approx(1.5**2)
    assert nn_product(phi) == pytest.approx(expected)


def test_nn_product_is_the_operator_kappa_multiplies(phi):
    # At lamb = 0 the action density is S/V = <phi^2> - 2 d kappa nn_product.
    kappa, d, V = 0.13, 3, np.prod(DIMS)
    action = Action(ScalarTerm(kappa=kappa, lamb=0.0))
    expected = np.mean(np.asarray(phi.F) ** 2) - 2 * d * kappa * nn_product(phi)

    assert action({"phi": phi}) / V == pytest.approx(expected)


@pytest.mark.parametrize("n", [(0, 0, 0), (1, 0, 0), (0, 1, 3), (-1, 2, 1)])
def test_structure_factor_is_the_fourier_transform(phi, n):
    F = np.asarray(phi.F)
    expected = np.abs(np.fft.fftn(F)[n]) ** 2 / F.size

    assert structure_factor(phi, n) == pytest.approx(expected)


def test_structure_factor_of_a_plane_wave(lattice):
    # phi = cos(p.x) puts V/4 into each of +-p and nothing elsewhere.
    n = (1, 1, 2)
    x = np.meshgrid(*[np.arange(L) for L in DIMS], indexing="ij")
    F = np.cos(sum(2 * np.pi * n_mu * x_mu / L for n_mu, x_mu, L in zip(n, x, DIMS)))
    wave = lat.LatticeField(lattice=lattice, F=F)
    V = np.prod(DIMS)

    assert structure_factor(wave, n) == pytest.approx(V / 4)
    assert structure_factor(wave, (1, 0, 0)) == pytest.approx(0, abs=1e-10)


def test_structure_factor_at_zero_momentum_is_V_m_squared(phi):
    V = np.prod(DIMS)
    assert structure_factor(phi, (0, 0, 0)) == pytest.approx(V * magnetization(phi) ** 2)


def test_structure_factor_needs_one_momentum_per_axis(phi):
    with pytest.raises(ValueError, match="momentum"):
        structure_factor(phi, (1, 0))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_correlator_definition(phi, axis):
    # C(t) = (1/V) sum_{t0} W(t0) W(t0 + t), W the slice sums along `axis`.
    F = np.asarray(phi.F)
    other = tuple(ax for ax in range(3) if ax != axis)
    W = F.sum(axis=other)
    T = DIMS[axis]
    expected = [sum(W[t0] * W[(t0 + t) % T] for t0 in range(T)) / F.size for t in range(T)]

    C = correlator(phi, axis=axis)
    assert C.shape == (T,)
    assert np.allclose(C, expected)


def test_correlator_sums_to_zero_momentum_structure_factor(phi):
    for axis in range(3):
        assert jnp.sum(correlator(phi, axis=axis)) == pytest.approx(
            structure_factor(phi, (0, 0, 0))
        )


def test_correlator_is_reflection_symmetric(phi):
    C = np.asarray(correlator(phi, axis=2))
    assert np.allclose(C[1:], C[1:][::-1])


def test_per_configuration_observables_are_jittable(phi):
    @jax.jit
    def measure(phi):
        return {
            "m": magnetization(phi),
            "nn": nn_product(phi),
            "Gp": structure_factor(phi, (1, 0, 0)),
            "C": correlator(phi),
        }

    out = measure(phi)
    assert out["m"] == pytest.approx(magnetization(phi))
    assert out["C"].shape == (DIMS[0],)


# --- ensemble estimators, on exact expectation values -------------------------

KAPPA = 0.15


def test_susceptibility_definition():
    m = np.random.default_rng(0).standard_normal(100) + 0.2
    V = 64
    assert susceptibility(m, volume=V) == pytest.approx(V * (np.mean(m**2) - np.mean(m) ** 2))


def test_broken_phase_susceptibility_is_susceptibility_of_abs_m():
    m = np.random.default_rng(1).standard_normal(100)
    V = 64
    assert susceptibility(np.abs(m), volume=V) == pytest.approx(
        V * (np.mean(m**2) - np.mean(np.abs(m)) ** 2)
    )


def test_binder_of_an_ordered_phase_is_two_thirds():
    m = np.array([0.7, -0.7] * 10)
    assert binder(m) == pytest.approx(2 / 3)


def test_xi_2nd_of_the_free_field():
    # Feed the exact moments: <m> = 0, <m^2> = G(0)/V, <G(p_min)> = G(p_min).
    # The lattice definition is then exactly sqrt(kappa / (1 - 2 d kappa)).
    dims = (12, 8, 8)
    V, L = np.prod(dims), dims[1]
    a = np.sqrt(free_field.chi_exact(dims, KAPPA) / V)
    m = np.array([a, -a])
    Gp = np.full(2, free_field.G_exact((0, 1, 0), dims, KAPPA))

    assert susceptibility(m, volume=V) == pytest.approx(free_field.chi_exact(dims, KAPPA))
    assert xi_2nd(m, Gp, volume=V, L=L) == pytest.approx(free_field.xi_2nd_exact(dims, KAPPA))


def test_xi_2nd_is_nan_where_it_does_not_exist():
    # xi exists only while chi >= <G(p_min)> > 0.  Outside that -- a uniform
    # cold start has G(p_min) = 0; a correlator that does not fall with
    # distance has chi < G(p_min) -- it should be a quiet nan, not an inf that
    # looks like a measurement, nor a RuntimeWarning per jackknife deletion.
    V, L = 16, 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cold = xi_2nd(np.ones(4), np.zeros(4), volume=V, L=L)  # chi = 0, Gp = 0
        spiky = xi_2nd(np.array([1.0, -1.0]), np.zeros(2), volume=V, L=L)  # Gp = 0
        flat = xi_2nd(np.array([0.1, -0.1]), np.ones(2), volume=V, L=L)  # chi < Gp
        missing = xi_2nd(np.array([np.nan, 1.0]), np.ones(2), volume=V, L=L)

    assert np.isnan(cold)
    assert np.isnan(spiky)
    assert np.isnan(flat)
    assert np.isnan(missing)


def test_xi_2nd_is_zero_in_the_infinite_mass_limit():
    # chi == <G(p_min)>: the correlator is a delta function in space, so xi = 0
    # exactly -- a legitimate value at the edge of the domain, not a nan.
    V, L = 16, 4
    m = np.array([0.25, -0.25])  # chi = V <m^2> = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert xi_2nd(m, np.ones(2), volume=V, L=L) == 0.0


def test_effective_mass_of_the_free_correlator_is_the_pole_mass():
    # The free C(t) is exactly cosh(m (t - T/2)), so m_eff is flat everywhere
    # except t = 0, where the periodic wrap breaks the cosh form.
    dims = (16, 4, 4)
    C = free_field.correlator_exact(dims, KAPPA)[None, :]  # a one-sample column
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the nan at t = 0 is by design
        m_eff = effective_mass(C)

    assert m_eff.shape == (16,)
    assert np.isnan(m_eff[0])
    assert np.allclose(m_eff[1:], free_field.pole_mass_exact(dims, KAPPA))


# --- end to end, on exact free-field samples ----------------------------------

FREE_DIMS = (12, 8, 8)
N_SAMPLES = 4000


@pytest.fixture(scope="module")
def free():
    lattice = lat.SquareLattice(st_dims=FREE_DIMS)
    configs = free_field.sample(jax.random.PRNGKey(42), FREE_DIMS, KAPPA, N_SAMPLES)
    return lattice, configs


def within(value, error, exact, nsigma=5):
    return np.all(np.abs(np.asarray(value) - exact) < nsigma * np.asarray(error))


def test_free_field_samples_target_the_lfox_action(free):
    # Equipartition: exp(-S) with S quadratic gives <S> = V/2.  This ties the
    # sampler's propagator to ScalarTerm's normalization.
    lattice, configs = free
    action = Action(ScalarTerm(kappa=KAPPA, lamb=0.0))
    S = free_field.measure(lambda phi: action({"phi": phi}), lattice, configs)

    assert within(S.mean(), S.std() / np.sqrt(N_SAMPLES), np.prod(FREE_DIMS) / 2)


@pytest.mark.parametrize("n", [(1, 0, 0), (0, 1, 0), (2, 1, 3)])
def test_free_field_structure_factor(free, n):
    lattice, configs = free
    G = free_field.measure(lambda phi: structure_factor(phi, n), lattice, configs)

    assert within(G.mean(), G.std() / np.sqrt(N_SAMPLES), free_field.G_exact(n, FREE_DIMS, KAPPA))


def test_free_field_correlator(free):
    lattice, configs = free
    C = free_field.measure(correlator, lattice, configs)

    assert C.shape == (N_SAMPLES, FREE_DIMS[0])
    assert within(
        C.mean(axis=0),
        C.std(axis=0) / np.sqrt(N_SAMPLES),
        free_field.correlator_exact(FREE_DIMS, KAPPA),
    )


def test_free_field_susceptibility_binder_and_xi(free):
    lattice, configs = free
    V, L = np.prod(FREE_DIMS), FREE_DIMS[1]
    m = free_field.measure(magnetization, lattice, configs)
    Gp = free_field.measure(lambda phi: structure_factor(phi, (0, 1, 0)), lattice, configs)

    chi, dchi = jackknife(susceptibility, m, volume=V)
    U, dU = jackknife(binder, m)
    xi, dxi = jackknife(xi_2nd, m, Gp, volume=V, L=L)

    assert within(chi, dchi, free_field.chi_exact(FREE_DIMS, KAPPA))
    assert within(U, dU, 0.0)  # m is exactly Gaussian
    assert within(xi, dxi, free_field.xi_2nd_exact(FREE_DIMS, KAPPA))
