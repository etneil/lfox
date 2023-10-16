from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import lfox.lattice
import lfox.fermions.spin as spin
from lfox.fermions.spin import Dirac4DFermionField as dirac


class WilsonDiracOp():

    def __init__(self, kappa):
        self.kappa = kappa

    @staticmethod
    def shift_fermion(psi, axis, shift):
        # Overload me to add gauge fields!
        return psi.nn_field(axis=axis, shift=shift)

    @partial(jax.jit, static_argnums=(0,))
    def Dslash(self, psi):
        Dpsi = psi.copy()
        Dpsi = Dpsi.zero_fill()

        for mu in range(psi.lattice.d):
            proj_minus = (spin.GammaMatrix['I'] - spin.GammaMatrix[mu])
            proj_plus = (spin.GammaMatrix['I'] + spin.GammaMatrix[mu])

            Dpsi += dirac._spin_product(proj_minus, self.shift_fermion(psi, mu, 1).F)
            Dpsi += dirac._spin_product(proj_plus, self.shift_fermion(psi, mu, -1).F)


        return Dpsi

    @partial(jax.jit, static_argnums=(0,))
    def op(self, psi):
        chi = psi.copy()
        chi -= self.kappa * self.Dslash(psi)

        return chi

        

