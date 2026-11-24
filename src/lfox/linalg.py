"""Linear algebra on fields.

A linear operator is a plain function from fields to fields -- a Dirac
operator is a closure over its bosonic background -- and nothing here assumes
more than linearity.
"""

import math

import jax
import jax.numpy as jnp


def dense(A, like):
    """The matrix of A on fields shaped like `like`, in `.F.ravel()` order.

    For small lattices: exact determinants and spectra.
    """
    shape = like.F.shape
    n = math.prod(shape)

    def column(e):
        return A(like.copy_new_F(e.reshape(shape))).F.ravel()

    return jax.vmap(column, out_axes=1)(jnp.eye(n, dtype=like.F.dtype))


def dagger(A, like):
    """The adjoint of a linear A, <a, A b> = <A^dagger a, b>, on fields like `like`.

    By transposition and conjugation, so it holds for any linear operator,
    not only gamma5-hermitian ones.
    """
    transpose = jax.linear_transpose(A, like)

    def A_dagger(a):
        (out,) = transpose(a.copy_new_F(a.F.conj()))
        return out.copy_new_F(out.F.conj())

    return A_dagger
