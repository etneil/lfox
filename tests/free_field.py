"""Exact samples of the free (lambda = 0) scalar field, for testing observables.

In the lfox hopping convention (`tests.phi4.ScalarTerm` at lamb = 0),

    S = sum_x [ phi_x^2 - 2 kappa sum_mu phi_x phi_{x+mu} ],

the theory is Gaussian with propagator

    G(p) = 1 / (2 (1 - 2 kappa sum_mu cos p_mu)),    p_mu = 2 pi n_mu / L_mu,

so exact, independent configurations cost one FFT each: colour white noise by
sqrt(G(p)).  Requires kappa < 1/(2d).  Everything here is closed-form, which is
what makes these the reference values the observables are tested against.
"""

import jax
import jax.numpy as jnp
import numpy as np

import lfox.lattice as lat


def propagator(st_dims, kappa):
    # G(p) on the FFT momentum grid.
    p = jnp.meshgrid(
        *[2 * jnp.pi * jnp.fft.fftfreq(L) for L in st_dims], indexing="ij"
    )
    return 1 / (2 * (1 - 2 * kappa * sum(jnp.cos(p_mu) for p_mu in p)))


def G_exact(n, st_dims, kappa):
    # G(p) at integer momentum n.
    cos_sum = sum(np.cos(2 * np.pi * n_mu / L) for n_mu, L in zip(n, st_dims))
    return 1 / (2 * (1 - 2 * kappa * cos_sum))


def chi_exact(st_dims, kappa):
    # Connected susceptibility, G(p = 0).
    return 1 / (2 * (1 - 2 * len(st_dims) * kappa))


def xi_2nd_exact(st_dims, kappa):
    # Second-moment correlation length.  On the free propagator the lattice
    # definition sqrt(G(0)/G(p_min) - 1) / (2 sin(pi/L)) is exactly this, for
    # any L.
    return np.sqrt(kappa / (1 - 2 * len(st_dims) * kappa))


def pole_mass_exact(st_dims, kappa):
    # G(p)^-1 = 2 kappa (2 cosh m - 2 cos p_t) at zero transverse momentum.
    return np.arccosh(1 + (1 - 2 * len(st_dims) * kappa) / (2 * kappa))


def correlator_exact(st_dims, kappa, axis=0):
    # C(t) = sum over transverse sites of G(x): the zero-transverse-momentum
    # line of G(p), transformed back along `axis`.
    G = np.asarray(propagator(st_dims, kappa))
    line = G[tuple(slice(None) if ax == axis else 0 for ax in range(len(st_dims)))]
    return np.fft.ifft(line).real


def sample(key, st_dims, kappa, n):
    # n independent configurations, shape (n, *st_dims).
    axes = tuple(range(1, len(st_dims) + 1))
    eta = jax.random.normal(key, (n, *st_dims))
    colored = jnp.fft.fftn(eta, axes=axes) * jnp.sqrt(propagator(st_dims, kappa))
    return jnp.fft.ifftn(colored, axes=axes).real


def measure(observable, lattice, configs):
    # Apply a per-configuration observable to a stack of raw arrays.
    wrapped = lambda F: observable(lat.LatticeField(lattice=lattice, F=F))
    return np.asarray(jax.vmap(wrapped)(configs), dtype=np.float64)
