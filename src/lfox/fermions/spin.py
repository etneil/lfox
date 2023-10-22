import numpy as np
import jax
import jax.numpy as jnp
from lfox.lattice import Lattice, LatticeField

Pauli_X = np.array([[0,1],[1,0]])
Pauli_Y = np.array([[0,-1j],[1j,0]])
Pauli_Z = np.array([[1,0],[0,-1]])

# DeGrand-Rossi gamma basis

Gamma_0 = np.array([
    [0, 0, 0, 1j],
    [0, 0, 1j, 0],
    [0, -1j, 0, 0],
    [-1j, 0, 0, 0],
])
Gamma_1 = np.array([
    [0, 0, 0, -1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
])
Gamma_2 = np.array([
    [0, 0, 1j, 0],
    [0, 0, 0, -1j],
    [-1j, 0, 0, 0],
    [0, 1j, 0, 0],
])
Gamma_3 = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])
Gamma_5 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1],
])

SigmaMatrix = {
    'X': Pauli_X,
    'Y': Pauli_Y,
    'Z': Pauli_Z,
    'I': np.eye(2),
}

GammaMatrix = {
    0: Gamma_0,
    1: Gamma_1,
    2: Gamma_2,
    3: Gamma_3,
    5: Gamma_5,
    'I': np.eye(4),
}

# TODO: add some algebraic tests to verify the above constants

# TODO: mark this all as 4D and/or generalize...

class Dirac4DFermionField(LatticeField):

    def __init__(self, lattice: Lattice, F=None, bc=None, indices=None):
        # "Indices" is being ignored, there is probably a better way...

        indices = (4,)

        super().__init__(lattice=lattice, indices=indices, F=F, bc=bc)

    def bilinear(self, other, spin_mat=None):
        new_fermion = self.copy()

        if spin_mat is None:
            new_fermion.F = self._inner_product(self.F, other.F)
        else:
            new_fermion.F = self._inner_spin_product(self.F, spin_mat, other.F)

        return new_fermion

    def conj(self):
        self.F = self.F.conj()
        return self
    
    def dot(self, other):
        return jnp.sum(self._inner_product(self.F, other.F))
    
    def norm_sq(self):
        return jnp.sum(self._inner_product(self.F, self.F))

    @staticmethod
    @jax.jit
    def _inner_product(psi_L, psi_R):
        return jnp.einsum('...i,...i', psi_L.conj(), psi_R)
    
    @staticmethod
    @jax.jit
    def _inner_spin_product(psi_L, spin_mat, psi_R):
        return jnp.einsum('...i,ij,...j', psi_L.conj(), spin_mat, psi_R)
    
    @staticmethod
    @jax.jit
    def _spin_product(spin_mat, psi):
        return jnp.einsum('ij,...j', spin_mat, psi)
    
jax.tree_util.register_pytree_node(
    Dirac4DFermionField,
    Dirac4DFermionField._tree_flatten,
    Dirac4DFermionField._tree_unflatten,
)