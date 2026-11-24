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


def _norm2(f):
    return jnp.vdot(f.F, f.F).real


def cg(A, b, x0=None, *, tol=1e-10, maxiter=1000):
    """Solve A x = b by conjugate gradient, for Hermitian positive-definite A.

    Returns `(x, info)`, with `info = {"iterations", "residual", "converged"}`
    and residual = |b - A x| / |b|.  Stops at `maxiter` without raising: a
    solve that did not converge says so in `info`.

    Differentiable by implicit differentiation (`lax.custom_linear_solve`),
    never through the iterations.  The transposed system is solved by CG as
    well: for complex Hermitian A it is conj(A), also Hermitian positive
    definite.  (Declaring the solve symmetric would be wrong for complex A,
    and would give a silently wrong gradient.)
    """

    def solve(matvec, b):
        bb = _norm2(b)
        x = b * 0.0 if x0 is None else x0
        r = b - matvec(x)

        def unconverged(state):
            _, _, _, rr, k = state
            return (rr > tol**2 * bb) & (k < maxiter)

        def step(state):
            x, r, p, rr, k = state
            Ap = matvec(p)
            alpha = rr / jnp.vdot(p.F, Ap.F).real
            x, r = x + alpha * p, r - alpha * Ap
            rr_new = _norm2(r)
            return x, r, r + (rr_new / rr) * p, rr_new, k + 1

        x, _, _, rr, k = jax.lax.while_loop(unconverged, step, (x, r, r, _norm2(r), 0))
        # b = 0 has the exact solution x = 0, with nothing to converge.
        residual = jnp.where(bb > 0, jnp.sqrt(rr / jnp.where(bb > 0, bb, 1.0)), 0.0)
        return x, {"iterations": k, "residual": residual, "converged": residual <= tol}

    return jax.lax.custom_linear_solve(A, b, solve, transpose_solve=solve, has_aux=True)
