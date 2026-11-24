"""Design tests for `lfox.linalg`, written before the implementation.  The
design is in `fermions_design.md`.

Linear operators are plain functions from fields to fields (a Dirac operator
is a closure over its bosonic background).  `lfox.linalg` works with any such
function, fermionic or not:

  * **`dense(A, like)`** is the matrix of A on the space of fields shaped like
    `like`, in `.F.ravel()` order (sites, then internal axes): for small
    lattices, exact determinants and spectra.
  * **`dagger(A, like)`** is the adjoint, <a, A b> = <A^dagger a, b>, for any
    linear A, by transposition and conjugation.  It does not rely on
    gamma5-hermiticity, which fails, for example, for a pseudoscalar Yukawa
    coupling i g5 pi.  Its output is a field like `like`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.fermions as fermions
import lfox.lattice as lat

L2 = lat.SquareLattice(st_dims=(4, 3))


@pytest.fixture
def linalg():
    import lfox.linalg

    return lfox.linalg


def _spinor(key, bc=(-1, 1)):
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    shape = (4, 3, 2)
    F = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    return lat.LatticeField(lattice=L2, F=F, axes={"spin": 2}, bc=bc)


def _inner(a, b):
    return jnp.vdot(a.F, b.F)


def _yukawa(kappa=0.15, g=0.7):
    # Wilson plus a pseudoscalar coupling i g5 pi: not gamma5-hermitian.
    pi = lat.LatticeField(lattice=L2, F=jax.random.normal(jax.random.PRNGKey(9), (4, 3)))
    g5 = fermions.gamma5(2)

    def D(psi):
        return fermions.wilson(psi, kappa) + 1j * g * pi * lat.einsum("ij,j->i", g5, psi)

    return D


# --- dense --------------------------------------------------------------------


def test_dense_of_a_scaling_is_a_multiple_of_the_identity(linalg):
    like = _spinor(0)
    M = np.asarray(linalg.dense(lambda f: 2.0 * f, like))
    assert M.shape == (24, 24)
    assert np.allclose(M, 2.0 * np.eye(24))


def test_dense_of_an_antiperiodic_hop():
    # On a 1-d antiperiodic lattice, (A psi)(x) = psi(x - 1), and psi(-1) is
    # -psi(3): a cyclic shift with a sign on the wrap.
    from lfox.linalg import dense

    like = lat.LatticeField(
        lattice=lat.SquareLattice(st_dims=(4,)), F=jnp.zeros(4, dtype=complex), bc=(-1,)
    )
    M = np.asarray(dense(lambda f: f.nn_field(0, 1), like))
    expected = np.zeros((4, 4))
    for x in range(1, 4):
        expected[x, x - 1] = 1
    expected[0, 3] = -1
    assert np.allclose(M, expected)


def test_dense_acts_like_the_operator_in_ravel_order(linalg):
    D = _yukawa()
    psi = _spinor(1)
    M = np.asarray(linalg.dense(D, psi))
    assert np.allclose(M @ np.asarray(psi.F).ravel(), np.asarray(D(psi).F).ravel())


# --- dagger -------------------------------------------------------------------


def test_dagger_is_the_adjoint(linalg):
    D = _yukawa()
    a, b = _spinor(2), _spinor(3)
    Ddag = linalg.dagger(D, a)

    assert jnp.isclose(_inner(a, D(b)), _inner(Ddag(a), b))
    out = Ddag(a)
    assert isinstance(out, lat.LatticeField)
    assert out.axes == a.axes
    assert out.bc == a.bc


def test_dagger_is_the_conjugate_transpose_matrix(linalg):
    D = _yukawa()
    like = _spinor(4)
    M = np.asarray(linalg.dense(D, like))
    Mdag = np.asarray(linalg.dense(linalg.dagger(D, like), like))
    assert np.allclose(Mdag, M.conj().T)


def test_dagger_does_not_assume_gamma5_hermiticity(linalg):
    # For the pseudoscalar coupling, g5 D g5 is not D^dagger -- but dagger is.
    D = _yukawa()
    like = _spinor(5)
    M = np.asarray(linalg.dense(D, like))
    G5 = np.kron(np.eye(12), np.asarray(fermions.gamma5(2)))
    assert not np.allclose(G5 @ M @ G5, M.conj().T)
    assert np.allclose(np.asarray(linalg.dense(linalg.dagger(D, like), like)), M.conj().T)


def test_dagger_of_dagger_is_the_operator(linalg):
    D = _yukawa()
    a = _spinor(6)
    twice = linalg.dagger(linalg.dagger(D, a), a)
    assert jnp.allclose(twice(a).F, D(a).F)


def test_dagger_is_jittable(linalg):
    D = _yukawa()
    a = _spinor(7)
    Ddag = linalg.dagger(D, a)
    assert jnp.allclose(jax.jit(Ddag)(a).F, Ddag(a).F)
