"""Design tests for solver diagnostics, from a term to the HMC monitor, written
before the implementation.  The design is in `fermions_design.md`
("Diagnostics: from the solve to the HMC monitor").

  * **A term's `total` may return `(value, info)`**, `info` a dict -- a
    pseudofermion term returns the `info` of its `cg` solve.  Terms that
    return a plain value are unchanged.
  * **`Action.S(fields, info=True)`** returns `(S, {term name: info})`, and
    **`Action.dS(fields, info=True)`** returns `(gradients, {term name: info})`.
    Terms without info do not appear.  The defaults are unchanged.
  * **HMC computes delta_H from totals**, H = S + sum 1/2 p.p, so it runs
    actions with nonlocal (total-only) terms -- which a pseudofermion term is.
  * **The HMC monitor** gets every solve of a trajectory, in order, under flat
    keys `"<term>.<key>"`: the evaluation of S at the start, each force
    evaluation (Nstep for leapfrog, 2 Nstep for Omelyan), and the evaluation
    of S at the proposed end.  So each key has length n_force + 2 per
    trajectory, and `evolve_many` stacks it to (traj, n_force + 2).  Reporting
    changes nothing about the trajectory itself.
  * **`Chain`** records these like any monitor entry, and warns (a
    `RuntimeWarning`) when any `"<term>.converged"` entry in a block is false.
    A failed solve is never turned into a rejection.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action, Term
from lfox.evolution import HMC, Chain, LeapfrogIntegrator, OmelyanIntegrator

L = lat.SquareLattice(st_dims=(4, 4))
MONITOR = {"delta_H", "P_acc", "accept"}


def _phi(key=0):
    return lat.LatticeField(lattice=L, F=jax.random.normal(jax.random.PRNGKey(key), (4, 4)))


def _laplacian(m2):
    # (m2 - Laplacian): positive definite, with smallest eigenvalue m2.
    def A(f):
        hops = f.nn_field(0, 1) + f.nn_field(0, -1)
        for mu in range(1, f.d()):
            hops = hops + f.nn_field(mu, 1) + f.nn_field(mu, -1)
        return (m2 + 2 * f.d()) * f - hops

    return A


class Gaussian(Term):
    """S = 1/2 phi . A^-1 phi, by a CG solve, reporting the solve's info."""

    m2: float
    maxiter: int = eqx.field(static=True, default=1000)

    def total(self, phi):
        from lfox.linalg import cg

        x, info = cg(_laplacian(self.m2), phi, maxiter=self.maxiter)
        return 0.5 * jnp.vdot(phi.F, x.F).real, info


class Probe(Term):
    """S = m2/2 sum phi^2 as a total, reporting sum phi^2: which config was seen."""

    m2: float

    def total(self, phi):
        s = jnp.sum(phi.F**2)
        return 0.5 * self.m2 * s, {"norm2": s}


class Mass(Term):
    m2: float

    def density(self, phi):
        return 0.5 * self.m2 * phi**2


class ZeroMode(Term):
    """Nonlocal, total only, and no info."""

    h: float

    def total(self, phi):
        return 0.5 * self.h * jnp.sum(phi.F) ** 2 / phi.F.size


class BadInfo(Term):
    def total(self, phi):
        return jnp.sum(phi.F**2), [1, 2]


# --- Action -------------------------------------------------------------------


def test_S_returns_info_on_request():
    phi = _phi()
    S = Action(Probe(m2=2.0))
    s2 = float(jnp.sum(phi.F**2))

    assert float(S.S({"phi": phi})) == pytest.approx(s2)
    value, info = S.S({"phi": phi}, info=True)
    assert float(value) == pytest.approx(s2)
    assert set(info) == {"Probe"}
    assert float(info["Probe"]["norm2"]) == pytest.approx(s2)


def test_terms_without_info_report_nothing():
    phi = _phi()
    S = Mass(m2=1.0) + ZeroMode(h=0.5)
    value, info = S.S({"phi": phi}, info=True)
    assert info == {}
    assert float(value) == pytest.approx(float(S.S({"phi": phi})))


def test_info_is_keyed_by_term_name():
    phi = _phi()
    S = Probe(m2=1.0, label="a") + Probe(m2=2.0, label="b") + Mass(m2=1.0)
    _, info = S.S({"phi": phi}, info=True)
    assert set(info) == {"a", "b"}


def test_dS_returns_info_on_request():
    from lfox.linalg import cg

    phi = _phi(1)
    S = Action(Gaussian(m2=0.5))
    grads, info = S.dS({"phi": phi}, info=True)

    assert jnp.allclose(grads["phi"].F, S.dS({"phi": phi})["phi"].F)
    # d/dphi of 1/2 phi . A^-1 phi is A^-1 phi.
    x, _ = cg(_laplacian(0.5), phi)
    assert jnp.allclose(grads["phi"].F, x.F)
    assert set(info["Gaussian"]) == {"iterations", "residual", "converged"}
    assert int(info["Gaussian"]["iterations"]) > 0
    assert bool(info["Gaussian"]["converged"])


def test_info_must_be_a_dict():
    with pytest.raises(ValueError, match="info"):
        Action(BadInfo()).S({"phi": _phi()})


# --- HMC ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "integrator, n_force",
    [(LeapfrogIntegrator(eps=0.1, Nstep=4), 4), (OmelyanIntegrator(eps=0.1, Nstep=4), 8)],
    ids=["leapfrog", "omelyan"],
)
def test_monitor_has_every_solve_of_a_trajectory(integrator, n_force):
    hmc = HMC(action=Gaussian(m2=0.5), integrator=integrator)
    _, monitor, _ = hmc.evolve({"phi": _phi(2)}, jax.random.PRNGKey(3))

    for key in ("iterations", "residual", "converged"):
        assert monitor[f"Gaussian.{key}"].shape == (n_force + 2,)
    assert np.all(np.asarray(monitor["Gaussian.converged"]))
    assert np.all(np.asarray(monitor["Gaussian.iterations"]) > 0)

    _, many, _ = hmc.evolve_many({"phi": _phi(2)}, jax.random.PRNGKey(3), traj=3)
    assert many["Gaussian.iterations"].shape == (3, n_force + 2)


def test_monitor_order_is_start_forces_end():
    # The first entry is S at the starting configuration, the last is S at the
    # proposal (a warmup kernel always accepts, so that is what it returns).
    phi = _phi(4)
    hmc = HMC(action=Probe(m2=1.0), integrator=LeapfrogIntegrator(eps=0.1, Nstep=5))
    fields, monitor, _ = hmc.as_warmup().evolve({"phi": phi}, jax.random.PRNGKey(5))

    norm2 = np.asarray(monitor["Probe.norm2"])
    assert norm2.shape == (7,)
    assert norm2[0] == pytest.approx(float(jnp.sum(phi.F**2)))
    assert norm2[-1] == pytest.approx(float(jnp.sum(fields["phi"].F ** 2)))


def test_actions_without_info_keep_the_monitor_as_it_was():
    hmc = HMC(action=Mass(m2=1.0), integrator=LeapfrogIntegrator(eps=0.1, Nstep=5))
    _, monitor, _ = hmc.evolve({"phi": _phi(6)}, jax.random.PRNGKey(7))
    assert set(monitor) == MONITOR


def test_hmc_runs_an_action_with_a_nonlocal_term():
    hmc = HMC(action=Mass(m2=1.0) + ZeroMode(h=0.5), integrator=LeapfrogIntegrator(eps=0.1, Nstep=5))
    fields, monitor, _ = hmc.evolve_many({"phi": _phi(8)}, jax.random.PRNGKey(9), traj=5)
    assert jnp.all(jnp.isfinite(fields["phi"].F))
    assert np.all(np.isfinite(np.asarray(monitor["delta_H"])))


def test_reporting_info_does_not_change_the_trajectory():
    # Probe is Mass written as a total that reports: same key, same chain.
    integrator = LeapfrogIntegrator(eps=0.1, Nstep=5)
    phi, key = _phi(10), jax.random.PRNGKey(11)
    a, mon_a, _ = HMC(action=Mass(m2=1.0), integrator=integrator).evolve_many({"phi": phi}, key, traj=4)
    b, mon_b, _ = HMC(action=Probe(m2=1.0), integrator=integrator).evolve_many({"phi": phi}, key, traj=4)

    assert jnp.allclose(a["phi"].F, b["phi"].F)
    assert np.allclose(mon_a["delta_H"], mon_b["delta_H"])
    assert np.array_equal(mon_a["accept"], mon_b["accept"])


# --- Chain --------------------------------------------------------------------


def _chain(maxiter=1000):
    hmc = HMC(action=Gaussian(m2=0.5, maxiter=maxiter), integrator=LeapfrogIntegrator(eps=0.1, Nstep=4))
    return Chain(evolver=hmc, seed=12, init_fields={"phi": _phi(13)})


def test_chain_records_solver_info():
    chain = _chain()
    chain.run(3)
    iterations = chain.monitor["Gaussian.iterations"]
    assert len(iterations) == 3
    assert all(len(row) == 6 for row in iterations)


def test_chain_warns_when_a_solve_does_not_converge():
    with pytest.warns(RuntimeWarning, match="converge"):
        _chain(maxiter=1).warmup(2)
    with pytest.warns(RuntimeWarning, match="converge"):
        _chain(maxiter=1).run(2)


def test_chain_is_quiet_when_solves_converge():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _chain().warmup(2).run(2)
