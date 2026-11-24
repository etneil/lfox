"""Euclidean gamma matrices: Hermitian, {g_mu, g_nu} = 2 delta_mu_nu.

`gammas(d)[mu]` goes with lattice axis mu.  Pauli matrices (spin size 2) in
d = 2, 3; DeGrand-Rossi (spin size 4) in d = 4.
"""

import jax.numpy as jnp
import numpy as np

_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

_DEGRAND_ROSSI = np.array([
    [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
])


def gammas(d):
    # Shape (d, s, s): the d gamma matrices.
    if d in (2, 3):
        return jnp.asarray(_PAULI[:d])
    if d == 4:
        return jnp.asarray(_DEGRAND_ROSSI)
    raise ValueError(f"gamma matrices are defined for d = 2, 3, 4, not d = {d}")


def gamma5(d):
    # Hermitian, squares to one, anticommutes with every gamma:
    # -i g0 g1 = sigma_3 in d = 2, g0 g1 g2 g3 = diag(1, 1, -1, -1) in d = 4.
    if d == 2:
        return jnp.diag(jnp.array([1.0, -1.0], dtype=complex))
    if d == 4:
        return jnp.diag(jnp.array([1.0, 1.0, -1.0, -1.0], dtype=complex))
    raise ValueError(f"gamma5 exists in d = 2 and 4, not d = {d}")
