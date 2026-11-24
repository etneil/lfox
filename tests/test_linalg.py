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
  * **`cg(A, b, x0=None, tol=1e-10, maxiter=1000)`** solves A x = b for
    Hermitian positive-definite A (M^dagger M) by conjugate gradient, and
    returns `(x, info)`: x is a field like b, and `info` holds `iterations`,
    `residual` (|b - A x| / |b|) and `converged` (residual <= tol).  It stops
    at `maxiter` without raising -- non-convergence is reported, not hidden.
    It is jittable, and differentiable by implicit differentiation, never
    through the iterations: for complex Hermitian A the transposed system is
    conj(A), not A, and getting that wrong gives a silently wrong gradient.
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


# --- cg -----------------------------------------------------------------------


def _mdagm(sigma, kappa=0.15):
    # M^dagger M for M = Wilson + a scalar Yukawa coupling: complex Hermitian PD.
    from lfox.linalg import dagger

    def M(psi):
        return fermions.wilson(psi, kappa) + sigma * psi

    Mdag = dagger(M, _spinor(0))
    return lambda psi: Mdag(M(psi))


def _sigma(key=11, scale=0.3):
    return lat.LatticeField(lattice=L2, F=scale * jax.random.normal(jax.random.PRNGKey(key), (4, 3)))


def _relative_residual(A, x, b):
    return float(jnp.linalg.norm((b - A(x)).F) / jnp.linalg.norm(b.F))


def test_cg_solves_a_hermitian_positive_definite_system(linalg):
    A, b = _mdagm(_sigma()), _spinor(12)
    x, info = linalg.cg(A, b, tol=1e-10)

    assert isinstance(x, lat.LatticeField)
    assert x.axes == b.axes
    assert x.bc == b.bc
    assert _relative_residual(A, x, b) < 1e-9
    exact = np.linalg.solve(np.asarray(linalg.dense(A, b)), np.asarray(b.F).ravel())
    assert np.allclose(np.asarray(x.F).ravel(), exact)


def test_cg_reports_its_diagnostics(linalg):
    A, b = _mdagm(_sigma()), _spinor(13)
    x, info = linalg.cg(A, b, tol=1e-10)

    assert set(info) == {"iterations", "residual", "converged"}
    assert 0 < int(info["iterations"]) < 1000
    assert bool(info["converged"])
    assert float(info["residual"]) <= 1e-10
    # The reported residual is the real one, up to rounding in the recursion.
    assert _relative_residual(A, x, b) < 10 * 1e-10


def test_cg_iteration_count_is_exact_on_three_eigenvalues(linalg):
    # In exact arithmetic CG converges in as many iterations as A has distinct
    # eigenvalues: a site-dependent scaling by 1, 2 or 3 takes exactly three.
    d = lat.LatticeField(lattice=L2, F=1.0 + jnp.arange(12).reshape(4, 3) % 3)
    _, info = linalg.cg(lambda psi: d * psi, _spinor(14), tol=1e-10)
    assert int(info["iterations"]) == 3
    assert bool(info["converged"])


def test_cg_stops_at_maxiter_and_says_so(linalg):
    d = lat.LatticeField(lattice=L2, F=1.0 + jnp.arange(12).reshape(4, 3) % 3)
    _, info = linalg.cg(lambda psi: d * psi, _spinor(15), tol=1e-10, maxiter=2)
    assert int(info["iterations"]) == 2
    assert not bool(info["converged"])
    assert float(info["residual"]) > 1e-10


def test_cg_starts_from_x0(linalg):
    A, b = _mdagm(_sigma()), _spinor(16)
    x, _ = linalg.cg(A, b, tol=1e-12)

    again, info = linalg.cg(A, b, x0=x, tol=1e-10)
    assert int(info["iterations"]) == 0  # already converged
    assert jnp.allclose(again.F, x.F)

    # A nearby guess (a random perturbation: a multiple of x would leave the
    # initial residual along b, the same Krylov space as starting from zero)
    # is much closer after the same few iterations.
    near = x + 1e-4 * _spinor(21)
    _, from_near = linalg.cg(A, b, x0=near, tol=1e-14, maxiter=3)
    _, from_zero = linalg.cg(A, b, tol=1e-14, maxiter=3)
    assert float(from_near["residual"]) < 1e-2 * float(from_zero["residual"])


def test_cg_of_zero_is_zero(linalg):
    b = _spinor(17).copy_new_F(jnp.zeros((4, 3, 2), dtype=complex))
    x, info = linalg.cg(_mdagm(_sigma()), b)
    assert jnp.all(x.F == 0)
    assert int(info["iterations"]) == 0
    assert bool(info["converged"])


def test_cg_is_jittable(linalg):
    b = _spinor(18)

    @jax.jit
    def solve(sigma):
        return linalg.cg(_mdagm(sigma), b, tol=1e-10)

    x, info = solve(_sigma())
    assert bool(info["converged"])
    assert _relative_residual(_mdagm(_sigma()), x, b) < 1e-9


@pytest.mark.parametrize("field", ["complex", "real"])
def test_cg_gradient_matches_finite_differences(linalg, field):
    # d/dsigma of b^dagger A(sigma)^-1 b.  jax.grad cannot differentiate a
    # while_loop, so this passing means cg differentiates implicitly.  The
    # complex case is the one a symmetric=True declaration gets silently
    # wrong (A^T = conj(A) for complex Hermitian A).
    if field == "complex":
        b = _spinor(19)

        def A(sigma):
            return _mdagm(sigma)
    else:
        b = lat.LatticeField(lattice=L2, F=jax.random.normal(jax.random.PRNGKey(19), (4, 3)))

        def A(sigma):  # a real, symmetric positive-definite operator
            return lambda f: (2.0 + sigma * sigma) * f - 0.2 * (f.nn_field(0, 1) + f.nn_field(0, -1))

    def S(sigma):
        x, _ = linalg.cg(A(sigma), b, tol=1e-13)
        return jnp.vdot(b.F, x.F).real

    sigma = _sigma()
    grad = jax.jit(jax.grad(S))(sigma)
    direction = jax.random.normal(jax.random.PRNGKey(20), (4, 3))
    h = 1e-5
    fd = (S(sigma.copy_new_F(sigma.F + h * direction))
          - S(sigma.copy_new_F(sigma.F - h * direction))) / (2 * h)
    assert float(jnp.sum(grad.F * direction)) == pytest.approx(float(fd), rel=1e-6)
