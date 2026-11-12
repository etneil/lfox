"""Standard observables for a real scalar field.

Two kinds, and the difference matters:

  * **Per-configuration** observables take a `LatticeField` and return a
    number (or a small array) for that one configuration.  They are
    jittable, and are what a `Chain` measurement function returns.
  * **Ensemble** estimators take *columns* of per-configuration samples
    (leading axis = Monte Carlo time) and return one estimate.  They are
    nonlinear in ensemble averages, so no single configuration has a value
    for them; get an error bar with `lfox.stats.jackknife`.

    m, Gp = chain.columns('m', 'Gp')
    chi, dchi = stats.jackknife(susceptibility, m, volume=V)
    xi, dxi = stats.jackknife(xi_2nd, m, Gp, volume=V, L=L)

Conventions: m is the per-site magnetization M/V.  The broken-phase
susceptibility V(<m^2> - <|m|>^2) is `susceptibility(np.abs(m), volume)`,
since |m|^2 = m^2.
"""

import jax.numpy as jnp
import numpy as np

# --- per configuration ------------------------------------------------------


def magnetization(phi):
    # m = (1/V) sum_x phi_x
    return jnp.mean(phi.F)


def nn_product(phi):
    # (1/(dV)) sum_{x,mu} phi_x phi_{x+mu}: the operator kappa multiplies,
    # averaged over the d axes.
    d = phi.d()
    return sum(jnp.mean(phi.F * phi.nn_field(ax).F) for ax in range(d)) / d


def structure_factor(phi, n):
    # G(p) = |sum_x e^{-ipx} phi_x|^2 / V at p_mu = 2 pi n_mu / L_mu, for an
    # integer momentum tuple `n` with one entry per axis.  Not connected: at
    # n = 0 this is V m^2.
    if len(n) != phi.d():
        raise ValueError(
            f"momentum {n} needs one entry per axis, {phi.d()} in all."
        )

    # exp(-ip.x) factorizes over axes, so contract one axis at a time: O(V)
    # work, and no complex copy of the whole field.
    ft = phi.F
    for n_mu in n:
        L = ft.shape[0]
        phase = jnp.exp(-2j * jnp.pi * n_mu * jnp.arange(L) / L)
        ft = jnp.tensordot(phase, ft, axes=1)

    return jnp.abs(ft) ** 2 / phi.F.size


def correlator(phi, axis=0):
    # Zero-momentum correlator along `axis`,
    #
    #   C(t) = (1/V) sum_{t0} W(t0) W(t0 + t),   W(t) = sum of phi on slice t,
    #
    # which is G(x) summed over the transverse directions.  Shape (L_axis,),
    # t = 0 .. L_axis - 1, and sum_t C(t) = V m^2.  Not connected.
    transverse = tuple(ax for ax in range(phi.d()) if ax != axis)
    W = jnp.fft.fft(jnp.sum(phi.F, axis=transverse))

    return jnp.fft.ifft(W * jnp.conj(W)).real / phi.F.size


# --- ensemble estimators ----------------------------------------------------


def susceptibility(m, volume):
    # chi = V (<m^2> - <m>^2)
    return volume * (np.mean(m**2) - np.mean(m) ** 2)


def binder(m):
    # U = 1 - <m^4> / (3 <m^2>^2): 0 for a Gaussian, 2/3 for an ordered phase.
    return 1 - np.mean(m**4) / (3 * np.mean(m**2) ** 2)


def xi_2nd(m, Gp, volume, L):
    # Second-moment correlation length,
    #
    #   xi = sqrt(chi / <G(p_min)> - 1) / (2 sin(pi / L)),
    #
    # with `Gp` a column of G(p_min) along an axis of extent L.  xi exists
    # only while chi >= <G(p_min)> > 0 (equality is the infinite-mass limit,
    # xi = 0); outside that, and for nan input, it is a quiet nan.
    chi, gp = susceptibility(m, volume), np.mean(Gp)
    if not chi >= gp > 0:
        return np.nan
    return np.sqrt(chi / gp - 1) / (2 * np.sin(np.pi / L))


def effective_mass(C):
    # m_eff(t) = arccosh((C(t-1) + C(t+1)) / (2 C(t))) on the mean of a
    # correlator column C, shape (n, T), periodic in t.  Returns shape (T,)
    # indexed by t, with nan at t = 0: C(-1) wraps to C(T-1), and the cosh
    # form does not hold across the wrap.
    C = np.mean(C, axis=0)
    ratio = (np.roll(C, 1) + np.roll(C, -1)) / (2 * C)
    ratio[0] = np.nan

    # Noise can push the ratio below 1; that is nan, not a warning.
    with np.errstate(invalid="ignore"):
        return np.arccosh(ratio)
