"""Dirac operators: linear functions of spinor fields."""

import jax.numpy as jnp

import lfox.lattice as lat
from lfox.fermions.gamma import gammas


def wilson(psi, kappa):
    """The Wilson operator in hopping form, acting on the "spin" axis:

        D psi = psi - kappa sum_mu [(1 - g_mu) psi(x+mu) + (1 + g_mu) psi(x-mu)]

    Other axes (color, flavor) pass through; hops pick up psi's boundary
    conditions.
    """
    d = psi.d()
    g = gammas(d)
    size = g.shape[-1]
    if dict(psi.axes).get("spin") != size:
        raise ValueError(
            f"wilson in d = {d} needs a 'spin' axis of size {size}; "
            f"this field has axes {psi.axes}"
        )

    I = jnp.eye(size)
    def hop(mu):
        # nn_field(mu, shift) gives site x the value from x - shift.
        forward = lat.einsum("ij,j->i", I - g[mu], psi.nn_field(mu, -1), axes="spin")
        backward = lat.einsum("ij,j->i", I + g[mu], psi.nn_field(mu, 1), axes="spin")
        return forward + backward

    # (Not sum(): its 0 start would be a constant added to an antiperiodic field.)
    total = hop(0)
    for mu in range(1, d):
        total = total + hop(mu)
    return psi - kappa * total
