"""Physics tests for O(N) phi^4: the first theory with an internal axis.

`tests.phi4.ScalarTerm` is written with `dot`, so one term serves every N, and
a field with no internal axes is N = 1.  These pin that it is the O(N) model,
and that HMC samples it:

  * N = 1 is the plain scalar, exactly.
  * the action is invariant under O(N) rotations of the field, and the force
    rotates with it.
  * autodiff agrees with the analytic force.
  * a uniform field of length c has the scalar action density at phi = c.
  * HMC satisfies <exp(-dH)> = 1, and at lamb = 0 it samples N independent
    free fields: equipartition <S> = N V / 2, and chi = N chi_free.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action
from lfox.evolution import HMC, Chain, LeapfrogIntegrator
from lfox.scalar import magnetization, susceptibility
from lfox.stats import jackknife
from tests import free_field
from tests.phi4 import ScalarTerm

DIMS = (4, 6, 4)
L = lat.SquareLattice(st_dims=DIMS)
N = 3
ACTION = Action(ScalarTerm(kappa=0.12, lamb=0.6))


def _vec(key, bc=(), n=N, scale=0.7):
    F = scale * jax.random.normal(jax.random.PRNGKey(key), DIMS + (n,))
    return lat.LatticeField(lattice=L, F=F, axes={"a": n}, bc=bc)


def _rotation(key, n=N):
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(key), (n, n)))
    return Q


def _rotate(R, phi):
    return lat.einsum("ij,j->i", R, phi)


# --- the action ---------------------------------------------------------------


def test_n_equals_one_is_the_plain_scalar():
    vec = _vec(0, n=1)
    plain = lat.LatticeField(lattice=L, F=vec.F[..., 0])

    assert ACTION({"phi": vec}) == pytest.approx(float(ACTION({"phi": plain})))
    assert jnp.allclose(
        ACTION.dS({"phi": vec})["phi"].F[..., 0], ACTION.dS({"phi": plain})["phi"].F
    )


@pytest.mark.parametrize("bc", [(), (-1, 1, 1), (1, 0, 1)], ids=["periodic", "anti", "dirichlet"])
def test_action_is_o_n_invariant(bc):
    phi = _vec(1, bc)
    R = _rotation(2)
    S = float(ACTION({"phi": phi}))

    assert float(ACTION({"phi": _rotate(R, phi)})) == pytest.approx(S, rel=1e-10)
    # ...and a matrix that is not a rotation does change it, so the check bites.
    stretch = jnp.diag(jnp.array([1.5, 1.0, 1.0]))
    assert float(ACTION({"phi": _rotate(stretch, phi)})) != pytest.approx(S, rel=1e-3)


def test_the_force_rotates_with_the_field():
    phi = _vec(3)
    R = _rotation(4)
    grad = ACTION.dS({"phi": phi})["phi"]
    rotated = ACTION.dS({"phi": _rotate(R, phi)})["phi"]
    assert jnp.allclose(rotated.F, _rotate(R, grad).F)


@pytest.mark.parametrize("bc", [(), (-1, 1, 1)], ids=["periodic", "anti"])
def test_force_matches_exact(bc):
    phi = _vec(5, bc)
    autodiff = ACTION.dS({"phi": phi})["phi"]
    exact = ACTION["ScalarTerm"].exact_gradient(phi)["phi"]

    assert autodiff.axes == exact.axes == phi.axes
    assert jnp.allclose(autodiff.F, exact.F)


def test_uniform_field_action():
    # A uniform field of length c, in any direction: per site
    # c^2 (1 - 2 kappa d) + lamb (c^2 - 1)^2, as for the scalar at phi = c.
    c, kappa, lamb, d = 1.3, 0.12, 0.6, len(DIMS)
    direction = jnp.array([1.0, -2.0, 2.0]) / 3.0
    phi = lat.LatticeField(
        lattice=L, F=jnp.broadcast_to(c * direction, DIMS + (N,)), axes={"a": N}
    )
    expected = np.prod(DIMS) * (c**2 * (1 - 2 * kappa * d) + lamb * (c**2 - 1) ** 2)

    assert float(ACTION({"phi": phi})) == pytest.approx(expected)


# --- HMC ----------------------------------------------------------------------


def test_on_detailed_balance():
    # Metropolis requires <exp(-dH)> = 1; +-10%, as for the plain scalar.
    hmc = HMC(action=ACTION, integrator=LeapfrogIntegrator(eps=0.1, Nstep=10))
    fields, rng_key = {"phi": _vec(6)}, jax.random.PRNGKey(7)
    fields, _, rng_key = hmc.as_warmup().evolve_many(fields, rng_key, traj=100)
    _, monitor, _ = hmc.evolve_many(fields, rng_key, traj=500)

    mean_exp = float(np.mean(np.exp(-np.asarray(monitor["delta_H"]))))
    assert 0.9 < mean_exp < 1.1, f"<exp(-dH)> = {mean_exp:.4f}, expected ~1.0"


def test_free_on_hmc_samples_n_independent_free_fields():
    # At lamb = 0 the O(N) model is N free fields, so HMC must reproduce
    # equipartition, <S> = (number of degrees of freedom) / 2 = N V / 2, and
    # chi = N chi_free.  This is an end-to-end check of the momenta, the
    # kinetic term and the force for a vector field.
    n, kappa, V = 2, 0.1, np.prod(DIMS)
    action = Action(ScalarTerm(kappa=kappa, lamb=0.0))
    hmc = HMC(action=action, integrator=LeapfrogIntegrator(eps=0.2, Nstep=5))
    chain = Chain(
        evolver=hmc,
        seed=8,
        init_fields={"phi": _vec(9, n=n)},
        measure=[lambda f, a: {"S": a(f), "m": magnetization(f["phi"])}],
    )
    chain.warmup(100)
    chain.run(2000)
    S, m = chain.columns("S", "m")
    assert m.shape == (2000, n)

    s, ds = jackknife(np.mean, S, bin_size=20)
    chi, dchi = jackknife(susceptibility, m, bin_size=20, volume=V)

    assert abs(s - n * V / 2) < 5 * ds, (s, ds)
    chi_exact = n * free_field.chi_exact(DIMS, kappa)
    assert abs(chi - chi_exact) < 5 * dchi, (chi, dchi, chi_exact)
