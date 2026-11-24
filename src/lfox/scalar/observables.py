"""Standard observables for a real scalar field, or an O(N) vector of them.

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

O(N): the plain scalar is N = 1.  A field with internal axes has a vector
magnetization m_a (the internal axes trail, as in `.F`); everything quadratic
contracts them, so G(p), C(t) and nn_product are sums over components.  The
estimators contract the trailing axes of their columns, so chi = V(<m.m> -
<m>.<m>) and the broken-phase form is `susceptibility(np.linalg.norm(m,
axis=-1), volume)`.  V is always the number of sites.
"""

import jax.numpy as jnp
import numpy as np

# --- per configuration ------------------------------------------------------


def _sites(phi):
    return tuple(range(phi.d()))


def magnetization(phi):
    # m_a = (1/V) sum_x phi_{x,a}: shape () for a plain scalar, (N,) for O(N).
    return jnp.mean(phi.F, axis=_sites(phi))


def nn_product(phi):
    # (1/(dV)) sum_{x,mu} phi_x . phi_{x+mu}: the operator kappa multiplies,
    # averaged over the d axes.
    d = phi.d()
    return sum(jnp.mean(phi.dot(phi.nn_field(ax)).F) for ax in range(d)) / d


def structure_factor(phi, n):
    # G(p) = sum_a |sum_x e^{-ipx} phi_{x,a}|^2 / V at p_mu = 2 pi n_mu / L_mu,
    # for an integer momentum tuple `n` with one entry per axis.  Not
    # connected: at n = 0 this is V m.m.
    if len(n) != phi.d():
        raise ValueError(
            f"momentum {n} needs one entry per axis, {phi.d()} in all."
        )

    # exp(-ip.x) factorizes over axes, so contract one axis at a time: O(V)
    # work, and no complex copy of the whole field.  What is left is the
    # internal axes, contracted in |.|^2.
    ft = phi.F
    for n_mu in n:
        L = ft.shape[0]
        phase = jnp.exp(-2j * jnp.pi * n_mu * jnp.arange(L) / L)
        ft = jnp.tensordot(phase, ft, axes=1)

    return jnp.sum(jnp.abs(ft) ** 2) / phi.lattice.volume


def correlator(phi, axis=0):
    # Zero-momentum correlator along `axis`,
    #
    #   C(t) = (1/V) sum_{t0} W(t0) . W(t0 + t),   W(t) = sum of phi on slice t,
    #
    # which is G(x) summed over the transverse directions (and the
    # components).  Shape (L_axis,), t = 0 .. L_axis - 1, and
    # sum_t C(t) = V m.m.  Not connected.
    transverse = tuple(ax for ax in _sites(phi) if ax != axis)
    W = jnp.fft.fft(jnp.sum(phi.F, axis=transverse), axis=0)
    C = jnp.fft.ifft(W * jnp.conj(W), axis=0).real

    return jnp.sum(C.reshape(C.shape[0], -1), axis=1) / phi.lattice.volume


# --- ensemble estimators ----------------------------------------------------


def _square(m):
    # m.m per sample, contracting any trailing (internal) axes of the column,
    # and N, the number of components.
    m = np.asarray(m)
    m2 = m**2
    return m2.reshape(len(m), -1).sum(axis=1), m2[0].size


def susceptibility(m, volume):
    # chi = V (<m.m> - <m>.<m>)
    m2, _ = _square(m)
    return volume * (np.mean(m2) - np.sum(np.mean(m, axis=0) ** 2))


def binder(m):
    # U = 1 - N/(N+2) <(m.m)^2> / <m.m>^2: 0 for a Gaussian, 2/(N+2) for an
    # ordered phase.  At N = 1 this is the usual 1 - <m^4> / (3 <m^2>^2).
    m2, N = _square(m)
    return 1 - N / (N + 2) * np.mean(m2**2) / np.mean(m2) ** 2


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
