"""Design tests for `lfox.fermions.PseudoFermion`, written before it.  The
design is in `fermions_design.md` ("Pseudofermions").

`PseudoFermion` is a `Term` for two degenerate flavors of any fermion, by one
pseudofermion `eta`.  A subclass supplies only the fermion matrix, as a linear
function of spinors closing over its bosonic fields:

    class YukawaFermion(PseudoFermion):
        kappa: float
        g: float

        def fermion_matrix(self, phi):
            return lambda psi: wilson(psi, self.kappa) + self.g * phi * psi

  * **Roles** are the parameters of `fermion_matrix`, in order, then `eta`.
    `eta` is sampled; the bosons are evolved.  `fermion_matrix` must name its
    fields explicitly, and may not call one `eta`.
  * **`total`** is eta^dag (M^dag M)^-1 eta, by a `cg` solve, returning the
    solve's info.  A nonlocal term: it has no density.
  * **`draw`** is the exact heatbath eta = M^dag xi, with xi a complex Gaussian
    of density exp(-xi^dag xi) -- so eta has covariance <eta eta^dag> = M^dag M
    and <eta eta^T> = 0.
  * **`gradient`** is supplied, with one solve, through
    `lfox.linalg.stationary_form`; it covers every role and returns the
    solve's info.  A subclass may override it with a hand-written force.  A
    subclass that overrides `total` must override `gradient` too: the
    inherited force would belong to the base class's action, not its own.
  * **Flavors** are copies on separate pseudofermions, by the usual
    `replicate` / `rebind` on `eta`.

The Yukawa-Wilson subclass `tests/pseudofermion.py` is checked on a lattice
small enough (16 complex components) to compare against the dense matrix of
M.  Together these pin what the identity int d eta exp(-S) ~ det(M^dag M)
rests on; sampling phi with that weight under HMC is left to integration
tests.
"""

import collections

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action
from lfox.fermions import PseudoFermion, gamma5, wilson
from lfox.linalg import cg, dagger, dense
from tests.pseudofermion import YukawaFermion

KAPPA, G = 0.2, 0.5
L = lat.SquareLattice(st_dims=(4, 2))
BC = (-1, 1)  # antiperiodic in time

CALLS = collections.Counter()


def _phi(key=0):
    return lat.LatticeField(lattice=L, F=jax.random.normal(jax.random.PRNGKey(key), (4, 2)))


def _eta(key=1):
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    F = jax.random.normal(k1, (4, 2, 2)) + 1j * jax.random.normal(k2, (4, 2, 2))
    return lat.LatticeField(lattice=L, F=F, axes={"spin": 2}, bc=BC)


def _M(phi):
    # The fermion matrix, written independently of the fixture.
    return lambda psi: wilson(psi, KAPPA) + G * phi * psi


def _S_dense(M, eta):
    Md = dense(M, eta)
    v = eta.F.ravel()
    return jnp.vdot(v, jnp.linalg.solve(Md.conj().T @ Md, v)).real


@pytest.fixture(scope="module")
def action():
    return Action(YukawaFermion(kappa=KAPPA, g=G))


# --- roles --------------------------------------------------------------------


def test_roles_are_the_fermion_matrix_signature_then_eta(action):
    assert YukawaFermion(kappa=KAPPA, g=G).field_names == ("phi", "eta")
    assert action.sampled_fields == ("eta",)
    assert action.evolved_fields == ("phi",)


def test_the_base_class_is_abstract():
    with pytest.raises(TypeError):
        PseudoFermion()

    class NoMatrix(PseudoFermion):
        kappa: float

    with pytest.raises(TypeError):
        NoMatrix(kappa=KAPPA)


def test_a_fermion_matrix_may_not_take_eta():
    with pytest.raises(TypeError):

        class Bad(PseudoFermion):
            def fermion_matrix(self, eta):
                return lambda psi: eta * psi


def test_a_fermion_matrix_must_name_its_fields():
    with pytest.raises(TypeError):

        class Bad(PseudoFermion):
            def fermion_matrix(self, *bosons):
                return lambda psi: psi


# --- action and heatbath ------------------------------------------------------


def test_action_is_eta_dagger_inverse_MdagM_eta(action):
    phi, eta = _phi(), _eta()
    S, info = action.S({"phi": phi, "eta": eta}, info=True)

    assert float(S) == pytest.approx(float(_S_dense(_M(phi), eta)), rel=1e-8)
    assert bool(info["YukawaFermion"]["converged"])
    assert int(info["YukawaFermion"]["iterations"]) > 0


def test_heatbath_has_covariance_MdagM(action):
    # eta = M^dag xi with <xi xi^dag> = 1 gives <eta eta^dag> = M^dag M; a
    # circular complex Gaussian also has <eta eta^T> = 0.  This is what pins
    # the noise normalization (a + ib)/sqrt(2), and M^dag rather than M.
    phi, eta = _phi(), _eta()
    fields = {"phi": phi, "eta": eta}
    K = 20_000
    keys = jax.random.split(jax.random.PRNGKey(2), K)
    draws = jax.vmap(lambda k: action.sample(fields, k)["eta"].F.ravel())(keys)

    Md = np.asarray(dense(_M(phi), eta))
    A = Md.conj().T @ Md
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


# --- force --------------------------------------------------------------------


def test_force_matches_the_dense_action(action):
    phi, eta = _phi(4), _eta(5)
    grads, info = action.dS({"phi": phi, "eta": eta}, info=True)

    assert set(grads) == {"phi"}
    exact = jax.grad(lambda F: _S_dense(_M(phi.copy_new_F(F)), eta))(phi.F)
    assert jnp.allclose(grads["phi"].F, exact, rtol=1e-7, atol=1e-10)
    assert bool(info["YukawaFermion"]["converged"])


def test_force_matches_finite_differences(action):
    phi, eta = _phi(6), _eta(7)
    delta = jax.random.normal(jax.random.PRNGKey(8), phi.F.shape)
    h = 1e-5

    def S(F):
        return float(action.S({"phi": phi.copy_new_F(F), "eta": eta}))

    fd = (S(phi.F + h * delta) - S(phi.F - h * delta)) / (2 * h)
    grad = action.dS({"phi": phi, "eta": eta})["phi"].F
    assert float(jnp.vdot(grad, delta)) == pytest.approx(fd, rel=1e-6)


def test_the_gradient_covers_eta_too(action):
    # Not needed by HMC, but `dS` may be asked for any field of the term.
    phi, eta = _phi(9), _eta(10)
    grads = action.dS({"phi": phi, "eta": eta}, wrt=("phi", "eta"))

    exact = jax.grad(lambda F: _S_dense(_M(phi), eta.copy_new_F(F)))(eta.F)
    assert jnp.allclose(grads["eta"].F, exact, rtol=1e-7, atol=1e-10)


def test_a_hand_written_gradient_takes_precedence():
    # The analytic force, dS/dphi(x) = -2 g Re[(M x)^dag(x) x(x)] with
    # x = (M^dag M)^-1 eta, is the pattern a hand-optimized subclass follows.
    class Analytic(YukawaFermion):
        def gradient(self, phi, eta):
            CALLS["Analytic.gradient"] += 1
            M = self.fermion_matrix(phi)
            x, info = cg(lambda v: dagger(M, eta)(M(v)), eta)
            local = jnp.sum((M(x).F.conj() * x.F).real, axis=-1)
            return {"phi": phi.copy_new_F(-2 * self.g * local)}, info

    phi, eta = _phi(11), _eta(12)
    fields = {"phi": phi, "eta": eta}
    before = CALLS["Analytic.gradient"]
    grads = Action(Analytic(kappa=KAPPA, g=G)).dS(fields)

    assert CALLS["Analytic.gradient"] > before
    default = Action(YukawaFermion(kappa=KAPPA, g=G)).dS(fields)
    assert jnp.allclose(grads["phi"].F, default["phi"].F, rtol=1e-7, atol=1e-10)


def test_overriding_total_requires_overriding_gradient():
    with pytest.raises(TypeError):

        class Rescaled(YukawaFermion):
            def total(self, phi, eta):
                S, info = super().total(phi=phi, eta=eta)
                return 2 * S, info

    # Both together is a consistent new term.
    class RescaledBoth(YukawaFermion):
        def total(self, phi, eta):
            S, info = super().total(phi=phi, eta=eta)
            return 2 * S, info

        def gradient(self, phi, eta):
            grads, info = super().gradient(phi=phi, eta=eta)
            return {k: 2 * v for k, v in grads.items()}, info

    RescaledBoth(kappa=KAPPA, g=G)


# --- more bosons, more flavors ------------------------------------------------


class TwoBoson(PseudoFermion):
    """Scalar and pseudoscalar Yukawa couplings: M = W + g phi + i h pi g5."""

    kappa: float
    g: float
    h: float

    def fermion_matrix(self, phi, pi):
        g5 = gamma5(2)
        return lambda psi: (
            wilson(psi, self.kappa)
            + self.g * phi * psi
            + 1j * self.h * pi * lat.einsum("ij,j->i", g5, psi)
        )


def test_a_fermion_matrix_of_two_bosons():
    term = TwoBoson(kappa=KAPPA, g=G, h=0.3)
    action = Action(term)
    phi, pi, eta = _phi(13), _phi(14), _eta(15)
    fields = {"phi": phi, "pi": pi, "eta": eta}

    assert term.field_names == ("phi", "pi", "eta")
    assert action.evolved_fields == ("phi", "pi")

    def S_dense(phi_F, pi_F):
        M = term.fermion_matrix(phi.copy_new_F(phi_F), pi.copy_new_F(pi_F))
        return _S_dense(M, eta)

    assert float(action.S(fields)) == pytest.approx(float(S_dense(phi.F, pi.F)), rel=1e-8)
    grads = action.dS(fields)
    exact_phi, exact_pi = jax.grad(S_dense, argnums=(0, 1))(phi.F, pi.F)
    assert jnp.allclose(grads["phi"].F, exact_phi, rtol=1e-7, atol=1e-10)
    assert jnp.allclose(grads["pi"].F, exact_pi, rtol=1e-7, atol=1e-10)


def test_four_flavors_are_two_replicated_pseudofermions():
    action = Action(YukawaFermion(kappa=KAPPA, g=G)).replicate(2, "eta")
    phi, eta_0, eta_1 = _phi(16), _eta(17), _eta(18)
    fields = {"phi": phi, "eta_0": eta_0, "eta_1": eta_1}

    assert action.sampled_fields == ("eta_0", "eta_1")
    expected = _S_dense(_M(phi), eta_0) + _S_dense(_M(phi), eta_1)
    assert float(action.S(fields)) == pytest.approx(float(expected), rel=1e-8)

    drawn = action.sample(fields, jax.random.PRNGKey(19))
    assert not jnp.allclose(drawn["eta_0"].F, drawn["eta_1"].F)
