# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pseudofermion HMC against an exact marginal
#
# A fermion determinant enters HMC through a pseudofermion: an auxiliary field
# $\eta$ with a Gaussian action whose integral reproduces the determinant.  Each
# trajectory then has two parts that must compose correctly:
#
# 1. a heatbath that draws $\eta$ exactly from $p(\eta \mid \phi)$, and
# 2. a molecular-dynamics trajectory in $\phi$ at fixed $\eta$, with a Metropolis
#    accept/reject step.
#
# Getting the composition wrong — refreshing $\eta$ at the wrong point in the
# trajectory, making the refresh depend on the accept decision, or reusing a
# random key between the heatbath and the accept step — produces a sampler
# that runs, accepts at a normal rate, and is subtly biased.
#
# This notebook checks the composition against a toy model whose answer is
# known exactly.  The action, per site, is
#
# $$
# S[\phi, \eta] = \sum_x \left[ \frac{m^2}{2} \phi_x^2
#     + \frac{\eta_x^2}{2 \, (1 + g \phi_x^2)} \right].
# $$
#
# The $\eta$ term plays the part of $\eta^\dagger (M^\dagger M)^{-1} \eta$ with
# $M^\dagger M \to 1 + g\phi^2$.  Integrating $\eta$ out gives, site by site,
#
# $$
# p(\phi) \propto e^{-m^2 \phi^2 / 2} \sqrt{1 + g \phi^2},
# $$
#
# the analogue of $\det(M^\dagger M)^{1/2}$.  The sites are independent, so every
# moment of $\phi$ is a one-dimensional integral.  The heatbath is exact too:
# $\eta = \sqrt{1 + g\phi^2}\,\xi$ with $\xi \sim N(0, 1)$.
#
# We run HMC in two regimes: a typical step size (high acceptance) and a coarse
# one (low acceptance).  A composition error that couples the $\eta$ refresh to
# the accept decision shows up most clearly when rejections are common.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import lfox.lattice as lat
from lfox.action import Action, Term
from lfox.evolution import HMC, Chain, LeapfrogIntegrator

# Double precision!
jax.config.update("jax_enable_x64", True)


# %% [markdown]
# ## The action
#
# Two terms.  The pseudofermion term declares `eta` as sampled and supplies its
# exact heatbath in `draw`; HMC gives momenta only to the remaining (evolved)
# field, `phi`.

# %%
class Mass(Term):
    m2: float

    def density(self, phi):
        return 0.5 * self.m2 * phi**2


class Pseudofermion(Term):
    g: float
    sampled = ("eta",)

    def density(self, phi, eta):
        return 0.5 * eta**2 / (1 + self.g * phi**2)

    def draw(self, key, phi):
        xi = jax.random.normal(key, phi.F.shape)
        return {"eta": phi.copy_new_F(jnp.sqrt(1 + self.g * phi.F**2) * xi)}


m2, g = 1.0, 4.0
action = Mass(m2=m2) + Pseudofermion(g=g)

print("evolved:", action.evolved_fields, " sampled:", action.sampled_fields)


# %% [markdown]
# ## Exact values
#
# Moments of the marginal by quadrature.  For comparison, dropping the
# pseudofermion would give a pure Gaussian with $\langle \phi^2 \rangle = 1/m^2$,
# so the determinant's effect is large enough to be resolved easily.
#
# Two more exact results.  The pseudofermion action has
# $\langle S_\mathrm{pf} \rangle / V = 1/2$ by equipartition, since it is
# Gaussian in $\eta$ at fixed $\phi$.  And any correct HMC satisfies
# $\langle e^{-\Delta H} \rangle = 1$.

# %%
def marginal(x):
    return np.exp(-0.5 * m2 * x**2) * np.sqrt(1 + g * x**2)


x = np.linspace(-12, 12, 200_001)
w = marginal(x)
Z = np.trapezoid(w, x)

exact = {
    "phi2": np.trapezoid(x**2 * w, x) / Z,
    "phi4": np.trapezoid(x**4 * w, x) / Z,
    "S_pf": 0.5,
    "exp(-dH)": 1.0,
}

print(f"<phi^2> = {exact['phi2']:.5f}   (Gaussian, no pseudofermion: {1 / m2:.5f})")
print(f"<phi^4> = {exact['phi4']:.5f}   (Gaussian, no pseudofermion: {3 / m2**2:.5f})")


# %% [markdown]
# ## Running the chain
#
# The initial `eta` is only a placeholder.  The chain's field dict has to
# contain every field of the action, but the heatbath overwrites `eta` before
# it is used.
#
# Observables are measured every `MEAS` trajectories, and configurations saved
# every `SAVE` for the histogram below.  Errors come from binning the
# measurements into `NBIN` bins, each much longer than the autocorrelation time
# in either regime.

# %%
dims = (8, 8, 8)
V = int(np.prod(dims))
lattice = lat.SquareLattice(st_dims=dims)
init_fields = {
    "phi": lat.LatticeField(lattice=lattice, F=jnp.zeros(dims)),
    "eta": lat.LatticeField(lattice=lattice, F=jnp.ones(dims)),
}

NTRAJ, MEAS, SAVE, NBIN = 200_000, 20, 200, 50

observables = {
    "phi2": [lambda f, S: float(jnp.mean(f["phi"].F ** 2)), MEAS],
    "phi4": [lambda f, S: float(jnp.mean(f["phi"].F ** 4)), MEAS],
    "S_pf": [lambda f, S: float(Action(S["Pseudofermion"])(f)) / V, MEAS],
}


def run(eps, Nstep, seed):
    hmc = HMC(action=action, integrator=LeapfrogIntegrator(eps=eps, Nstep=Nstep))
    chain = Chain(
        evolver=hmc,
        seed=seed,
        init_fields=init_fields,
        observables=observables,
        save_freq=SAVE,
    )
    chain.warmup(500)
    chain.run(NTRAJ)

    return chain


def binned(samples, nbin=NBIN):
    # Mean and standard error from `nbin` equal bins.
    samples = np.asarray(samples)
    n = len(samples) // nbin * nbin
    bins = samples[:n].reshape(nbin, -1).mean(axis=1)

    return bins.mean(), bins.std(ddof=1) / np.sqrt(nbin)


# %%
# %%time
regimes = {
    "typical": run(eps=0.15, Nstep=7, seed=1),
    "coarse": run(eps=0.30, Nstep=5, seed=2),
}


# %% [markdown]
# ## Results
#
# Each measured value against its exact counterpart, with the pull
# (measured − exact) / error.  For a correct sampler the pulls should look like
# draws from $N(0, 1)$: mostly within $\pm 2$, and essentially never beyond
# $\pm 3$.
#
# $\langle e^{-\Delta H} \rangle$ is the most sensitive single check, since it
# needs the $\eta$ in $H$ at the start and end of the trajectory to be the same
# correctly drawn field.  At low acceptance it is dominated by rare trajectories
# with large negative $\Delta H$, so its error bar is wider and less Gaussian
# than the others.

# %%
def measured(chain):
    out = {name: binned(chain.obs_chain[name]) for name in observables}
    dH = np.asarray(chain.monitor["delta_H"][-chain.traj :])
    out["exp(-dH)"] = binned(np.exp(-dH))

    return out


print(f"{'':10s} {'observable':10s} {'measured':>20s} {'exact':>10s} {'pull':>7s}")
for label, chain in regimes.items():
    print(f"{label:10s} acceptance = {chain.acceptance():.2f}")
    for name, (mean, err) in measured(chain).items():
        pull = (mean - exact[name]) / err
        print(f"{'':10s} {name:10s} {mean:12.5f} ± {err:.5f} {exact[name]:10.5f} {pull:+7.1f}")


# %% [markdown]
# ## The full distribution
#
# The moments test a few numbers.  The histogram tests the whole marginal,
# including the tails where $\sqrt{1 + g\phi^2}$ matters most.  At $g = 4$ the
# determinant splits the single Gaussian peak in two.  The Gaussian without the
# pseudofermion is shown for scale.
#
# A percent-level bias is invisible at this resolution; the table above is the
# sensitive test, and the histogram is a check on gross errors.

# %%
fig, axes = plt.subplots(1, len(regimes), figsize=(11, 4), sharey=True)
x_plot = np.linspace(-5, 5, 400)
gauss = np.exp(-0.5 * m2 * x_plot**2) * np.sqrt(m2 / (2 * np.pi))

for ax, (label, chain) in zip(axes, regimes.items()):
    samples = np.concatenate([np.ravel(f.F) for f in chain.field_chain["phi"][1:]])
    ax.hist(samples, bins=120, range=(-5, 5), density=True, alpha=0.5, label="HMC")
    ax.plot(x_plot, marginal(x_plot) / Z, color="k", label="exact marginal")
    ax.plot(x_plot, gauss, color="k", ls=":", label="no pseudofermion")
    ax.set_title(f"{label} (acceptance {chain.acceptance():.2f})")
    ax.set_xlabel(r"$\phi$")

axes[0].set_ylabel(r"$p(\phi)$")
axes[0].legend()
plt.tight_layout()
