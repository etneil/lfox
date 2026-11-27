# ---
# jupyter:
#   jupytext:
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
# ### Yukawa theory test
#
# This notebook reproduces some basic numerical results by G. Kanwar in the reference "Simulating Quantum Field Theories on the Lattice" (2017).  Kanwar studies a simple Yukawa theory in $d=2$ dimensions, including $N_f$ identical fermions to study scaling in $1/N_f$, which can be predicted analytically.
#
# The lattice action of the theory is written as $S = S_{\phi} + S_{\psi}$, where:
#
# $$
# S_\phi = -2J \sum_{x,\mu} \phi(x) \phi(x+\mu) + \lambda (\phi(x)^2 - 1)^2 + \phi(x)^2,
# $$
# and
# $$
# S_\psi = \sum_{i=1}^{N_f} \bar{\psi}_i(x) D(x,y) \psi_i(y),
# $$
# where the Dirac operator is given by
# $$
# D(x,y) = \delta_{xy} \mathbf{1} (g\phi(x) + m_\psi) + \frac{1}{2a} \sum_\mu \gamma^\mu (\delta_{x+\mu, y} - \delta{x-\mu, y}).
# $$
#
# In practice, the lattice action is implemented using pseudofermions $\eta$, so that $S_{\psi}$ is replaced with $S_\eta$, where
# $$
# S_\eta = \eta^\dagger (D^\dagger D)^{-1} \eta.
# $$
#
# In $d=2$, the $\gamma^\mu$ matrices are just the usual Pauli matrices $\{ \sigma_x, \sigma_y\}$.  The theory has four free parameters: the scalar hopping term $J$, the $\phi^4$ coupling $\lambda$, the Yukawa coupling $g$, and the fermion mass $m_\psi$.
#
#
#

# %%
import lfox
import lfox.lattice as lat

from lfox.action import Action, Term
from lfox.fermions import wilson, PseudoFermion
import lfox.stats as lstats

import jax
import jax.numpy as jnp
import numpy as np
from lfox.evolution import HMC, Chain
from lfox.evolution.integrators import Integrator, Omelyan, Level
from lfox.linalg import dagger, cg, dense


import matplotlib.pyplot as plt
import lsqfit
import gvar as gv
import tqdm

jax.config.update("jax_enable_x64", True)


# %%
## Define the action terms
class ScalarAction(Term):
    
    J: float
    lamb: float

    def density(self, phi):
        S = phi**2

        for ax in range(phi.d()):
            S -= 2 * self.J * phi.dot(phi.nn_field(ax))

        S += self.lamb * (phi.dot(phi) - 1)**2

        return S

class YukawaFermion(PseudoFermion):
    g: float
    m: float

    def fermion_matrix(self, phi):
        kappa = 1/(2*(self.m+phi.d()))

        return lambda psi: wilson(psi, kappa) + 2 * kappa * self.g * psi * phi


class ExactFermion(Term):

    g: float
    m: float

    def DiracOp(self, phi):
        kappa = 1/(2*(self.m+phi.d()))

        return lambda psi: wilson(psi, kappa) + 2 * kappa * self.g * psi * phi

    def total(self, phi):
        spinor_like = lat.LatticeField(
            lattice=phi.lattice,
            F=jnp.zeros(phi.F.shape + (phi.d(),), complex),
            axes={"spin": 2},
            bc=(-1,1),
        )
        
        M = dense(self.DiracOp(phi), spinor_like)
        return -2 * jnp.linalg.slogdet(M)[1]
        


# %%
## Observables

def _sites(phi):
    return tuple(range(phi.d()))

def obs_m(phi):
    return jnp.abs(jnp.mean(phi.F, axis=_sites(phi)))

def obs_s(phi):
    eta_s = (-1)**jnp.sum(jnp.indices(phi.dims()),axis=0)

    return jnp.abs(jnp.mean(phi.F * eta_s, axis=_sites(phi)))



# %%
d = 2
Lat8 = lat.SquareLattice(st_dims=((8,)*d))
phi8 = lat.LatticeField(lattice=Lat8, F=1, bc=(1,1))
eta8 = lat.LatticeField(lattice=Lat8, F=1+0j, axes={"spin": 2}, bc=(-1,1))
psi8 = lat.LatticeField(lattice=Lat8, F=1+0j, axes={"spin": 2}, bc=(-1,1))

print(phi8)
print(eta8)
print(obs_m(phi8))
print(obs_s(phi8))

# %%
## Initial HMC test

S = ScalarAction(J=1, lamb=1) + YukawaFermion(g=0.5, m=1)

hmc_test = HMC(
    action = S,
    integrator = Integrator(tau=1.0, levels=[
        Level(Omelyan, Nstep=4, terms=('YukawaFermion')),
        Level(Omelyan, Nstep=6, terms=('ScalarAction')),
    ])
)

def scalar_meas(fields, action):
    return {
        'm': obs_m(fields['phi']),
        's': obs_s(fields['phi']),
    }

chain_test = Chain(
    evolver = hmc_test,
    seed = 59195,
    init_fields = {'phi': phi8, 'eta': eta8},
    measure=[(scalar_meas,10)],
    save_freq=100,
)


# %%
# %%prun -l 20

chain_test.warmup(1000)
chain_test.run(20000)


# %%
plt.plot(chain_test.column('m'))
plt.plot(chain_test.column('s'))

# %%
print(jnp.mean(chain_test.column('m'))) 
print(jnp.mean(chain_test.column('s')))

# %%
## Check HMC monitoring

plt.plot(chain_test.monitor['delta_H'])
plt.show()

plt.plot(chain_test.monitor['YukawaFermion.residual'])

print("Average accept rate: ", np.mean(chain_test.monitor['accept']))
print("exp(-ΔΗ) = ", np.mean(np.exp(-np.asarray(chain_test.monitor['delta_H']))))


# %%
## Secondary chain test: exact fermion determinant

## Initial HMC test

S = ScalarAction(J=1, lamb=1) + ExactFermion(g=0.5, m=1)

hmc_test = HMC(
    action = S,
    integrator = Integrator(tau=1.0, levels=[
        Level(Omelyan, Nstep=4, terms=('ExactFermion')),
        Level(Omelyan, Nstep=6, terms=('ScalarAction')),
    ])
)

def scalar_meas(fields, action):
    return {
        'm': obs_m(fields['phi']),
        's': obs_s(fields['phi']),
    }

chain_test = Chain(
    evolver = hmc_test,
    seed = 59195,
    init_fields = {'phi': phi8},
    measure=[(scalar_meas,10)],
    save_freq=100,
)


# %%
# %%prun -l 20

chain_test.warmup(1000)
chain_test.run(20000)


# %%
plt.plot(chain_test.column('m'))
plt.plot(chain_test.column('s'))

# %%
print(jnp.mean(chain_test.column('m'))) 
print(jnp.mean(chain_test.column('s')))

# %%
