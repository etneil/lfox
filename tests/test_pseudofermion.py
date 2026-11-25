"""Design tests for the pseudofermion test fixture, written before it.  The
design is in `fermions_design.md` ("Pseudofermions").

`tests/pseudofermion.py` provides `Pseudofermion(kappa, g)`, two degenerate
flavors of a Yukawa-Wilson fermion, M(phi) psi = wilson(psi, kappa) + g phi psi,
by one pseudofermion `eta` of M^dag M:

  * **`total`** is eta^dag (M^dag M)^-1 eta, by a `cg` solve, returning the
    solve's info.  A nonlocal term: it has no density.
  * **`draw`** is the exact heatbath eta = M^dag xi, with xi a complex Gaussian
    of density exp(-xi^dag xi) -- so eta has covariance <eta eta^dag> = M^dag M
    and <eta eta^T> = 0.  `eta` is sampled; `phi` is evolved.
  * **The force** is autodiff through the solve.

Everything is checked on a lattice small enough (16 complex components) to
compare against the dense matrix of M.  Together these pin what the identity
int d eta exp(-S) ~ det(M^dag M) rests on; sampling phi with that weight under
HMC is left to integration tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action
from lfox.fermions import wilson
from lfox.linalg import dense
from tests.pseudofermion import Pseudofermion

KAPPA, G = 0.2, 0.5
L = lat.SquareLattice(st_dims=(4, 2))
BC = (-1, 1)  # antiperiodic in time


def _phi(key=0):
    return lat.LatticeField(lattice=L, F=jax.random.normal(jax.random.PRNGKey(key), (4, 2)))


def _eta(key=1):
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    F = jax.random.normal(k1, (4, 2, 2)) + 1j * jax.random.normal(k2, (4, 2, 2))
    return lat.LatticeField(lattice=L, F=F, axes={"spin": 2}, bc=BC)


def _MdagM(phi, eta):
    # The dense M^dag M, written independently of the fixture.
    M = dense(lambda psi: wilson(psi, KAPPA) + G * phi * psi, eta)
    return M.conj().T @ M


def _S_dense(phi, eta):
    v = eta.F.ravel()
    return jnp.vdot(v, jnp.linalg.solve(_MdagM(phi, eta), v)).real


@pytest.fixture(scope="module")
def action():
    return Action(Pseudofermion(kappa=KAPPA, g=G))


def test_eta_is_sampled_and_phi_evolved(action):
    assert action.sampled_fields == ("eta",)
    assert action.evolved_fields == ("phi",)


def test_action_is_eta_dagger_inverse_MdagM_eta(action):
    phi, eta = _phi(), _eta()
    S, info = action.S({"phi": phi, "eta": eta}, info=True)

    assert float(S) == pytest.approx(float(_S_dense(phi, eta)), rel=1e-8)
    assert bool(info["Pseudofermion"]["converged"])
    assert int(info["Pseudofermion"]["iterations"]) > 0


def test_heatbath_has_covariance_MdagM(action):
    # eta = M^dag xi with <xi xi^dag> = 1 gives <eta eta^dag> = M^dag M; a
    # circular complex Gaussian also has <eta eta^T> = 0.  This is what pins
    # the noise normalization (a + ib)/sqrt(2), and M^dag rather than M.
    phi, eta = _phi(), _eta()
    fields = {"phi": phi, "eta": eta}
    K = 20_000
    keys = jax.random.split(jax.random.PRNGKey(2), K)
    draws = jax.vmap(lambda k: action.sample(fields, k)["eta"].F.ravel())(keys)

    A = np.asarray(_MdagM(phi, eta))
    C = np.asarray(draws.T @ draws.conj()) / K
    C_pseudo = np.asarray(draws.T @ draws) / K

    # Each entry's statistical error is about sqrt(A_ii A_jj / K).
    scale = np.sqrt(np.outer(np.diag(A).real, np.diag(A).real) / K)
    assert np.max(np.abs(C - A) / scale) < 5
    assert np.max(np.abs(C_pseudo) / scale) < 5


def test_heatbath_draws_eta_like_the_field_it_replaces(action):
    eta = _eta()
    drawn = action.sample({"phi": _phi(), "eta": eta}, jax.random.PRNGKey(3))["eta"]

    assert drawn.axes == eta.axes
    assert drawn.bc == eta.bc
    assert drawn.F.shape == eta.F.shape
    assert jnp.iscomplexobj(drawn.F)


def test_force_matches_the_dense_action(action):
    phi, eta = _phi(4), _eta(5)
    grads, info = action.dS({"phi": phi, "eta": eta}, info=True)

    assert set(grads) == {"phi"}
    exact = jax.grad(lambda F: _S_dense(phi.copy_new_F(F), eta))(phi.F)
    assert jnp.allclose(grads["phi"].F, exact, rtol=1e-7, atol=1e-10)
    assert bool(info["Pseudofermion"]["converged"])


def test_force_matches_finite_differences(action):
    phi, eta = _phi(6), _eta(7)
    delta = jax.random.normal(jax.random.PRNGKey(8), phi.F.shape)
    h = 1e-5

    def S(F):
        return float(action.S({"phi": phi.copy_new_F(F), "eta": eta}))

    fd = (S(phi.F + h * delta) - S(phi.F - h * delta)) / (2 * h)
    grad = action.dS({"phi": phi, "eta": eta})["phi"].F
    assert float(jnp.vdot(grad, delta)) == pytest.approx(fd, rel=1e-6)
