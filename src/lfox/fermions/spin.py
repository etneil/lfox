import numpy as np
from lfox.lattice import Lattice, LatticeTensorField

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
}

GammaMatrix = {
    0: Gamma_0,
    1: Gamma_1,
    2: Gamma_2,
    3: Gamma_3,
    5: Gamma_5,
}

# TODO: add some algebraic tests to verify the above constants

# TODO: mark this all as 4D and/or generalize...

class DiracFermionField(LatticeTensorField):

    def __init__(self, lattice: Lattice, F=None, bc=None):
        indices = [
            ('spin', 4),
        ]

        super().__init__(lattice=lattice, indices=indices, F=F, bc=bc)

