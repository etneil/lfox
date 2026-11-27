"""Design tests for `MultiChain`, written before the implementation.

`MultiChain` runs a batch of independent Markov chains in lockstep, as one
vmapped computation.  The members may differ in any traced parameter -- a
coupling, the step size `eps` -- for a parameter scan, or only in their seeds,
for replica streams at one parameter set.

  * **Construction**: `MultiChain(evolvers, seeds, init_fields, measure=None,
    save_freq=1)`.  `evolvers` and `seeds` are sequences of equal length B, one
    per member.  The evolvers are stacked leaf by leaf, so they must share
    their static structure (the same terms, the same `Nstep`, ...); a mismatch
    raises a ValueError at construction.  `init_fields` is one dict, shared
    by every member, or a list of B dicts.
  * **Member i is `Chain(evolvers[i], seeds[i], init_fields[i])`**: the same
    monitor, measurements and configurations, to round-off.  Batching is a
    performance knob, never physics.
  * **Everything per member gets a leading batch axis**: `monitor[key]` is
    (B, T, ...), `column(name)` is (B, T, ...), `acceptance()` is (B,), and
    `fields` and each `field_chain` entry hold stacked fields, F of shape
    (B, *lattice shape).  Trajectory bookkeeping (`traj`, `warmup_traj`,
    `traj_chain`, `column_traj`) is shared, since the members run in lockstep.
    A stacked field is not a field of the lattice it names (code that reduces
    over its sites silently reduces over the wrong axes), so
    **`member_fields(i)`** returns member i's current fields as ordinary
    `LatticeField`s.
  * **Measurement functions are written for one chain**, f(fields, action),
    and each member's is called with its own fields and its own action.
  * **Non-convergence** warns as for `Chain`, naming the members that failed.
  * **Duplicate seeds warn** (a UserWarning naming the seed), even when the
    members' parameters differ.

Within a batch the cost of every member is that of the most expensive one (a
CG solve runs until the slowest member converges); members of very different
cost belong in separate batches.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action, Term
from lfox.evolution import HMC, Chain, LeapfrogIntegrator, MultiChain
from lfox.linalg import cg
from tests.phi4 import ScalarTerm

KAPPAS = (0.15, 0.18, 0.21)
EPS = (0.1, 0.08, 0.12)
SEEDS = (11, 12, 13)
B = len(KAPPAS)


def _hmc(kappa=0.18, eps=0.1, Nstep=5):
    return HMC(
        action=Action(ScalarTerm(kappa=kappa, lamb=1.3282)),
        integrator=LeapfrogIntegrator(eps=eps, Nstep=Nstep),
    )


@pytest.fixture
def evolvers():
    return [_hmc(kappa=k, eps=e) for k, e in zip(KAPPAS, EPS)]


def mag(fields, action):
    return {'m': jnp.mean(fields['phi'].F)}


def profile(fields, action):
    return {'C': jnp.mean(fields['phi'].F, axis=(1, 2))}


def coupling(fields, action):
    return {'kappa': action['ScalarTerm'].kappa}


def _random_phi(lattice, key):
    return lat.LatticeField(lattice=lattice, F=jax.random.normal(jax.random.PRNGKey(key), (4, 4, 4)))


# --- Construction -------------------------------------------------------------


def test_multichain_starts_every_member_from_a_shared_start(evolvers, phi4):
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4})

    assert len(mc) == B
    assert mc.traj == 0
    assert mc.traj_chain == [0]
    assert mc.fields['phi'].F.shape == (B, 4, 4, 4)
    assert np.all(mc.fields['phi'].F == phi4.F)


def test_multichain_takes_a_start_per_member(evolvers, lat4):
    starts = [{'phi': _random_phi(lat4, i)} for i in range(B)]
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields=starts)

    for i in range(B):
        assert np.array_equal(mc.fields['phi'].F[i], starts[i]['phi'].F)


def test_multichain_needs_one_seed_per_evolver(evolvers, phi4):
    with pytest.raises(ValueError, match="seeds"):
        MultiChain(evolvers=evolvers, seeds=SEEDS[:2], init_fields={'phi': phi4})


def test_multichain_needs_one_start_per_evolver(evolvers, phi4):
    with pytest.raises(ValueError, match="init_fields"):
        MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields=[{'phi': phi4}] * 2)


def test_multichain_needs_at_least_one_member(phi4):
    with pytest.raises(ValueError):
        MultiChain(evolvers=[], seeds=[], init_fields={'phi': phi4})


def test_multichain_rejects_evolvers_of_different_structure(phi4):
    # Nstep is static: members that differ in it cannot share one computation.
    evolvers = [_hmc(Nstep=5), _hmc(Nstep=6)]
    with pytest.raises(ValueError, match="structure"):
        MultiChain(evolvers=evolvers, seeds=(1, 2), init_fields={'phi': phi4})


def test_multichain_rejects_starts_of_different_shape(phi4):
    small = lat.LatticeField(lattice=lat.SquareLattice(st_dims=(2, 2, 2)), F=jnp.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="structure"):
        MultiChain(
            evolvers=[_hmc(), _hmc()],
            seeds=(1, 2),
            init_fields=[{'phi': phi4}, {'phi': small}],
        )


def test_multichain_rejects_mismatched_field_names(evolvers, phi4):
    with pytest.raises(ValueError, match="do not match"):
        MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'psi': phi4})


def test_multichain_rejects_measure_not_returning_a_dict(evolvers, phi4):
    def bare(fields, action):
        return jnp.mean(fields['phi'].F)

    with pytest.raises(TypeError, match="dict"):
        MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, measure=[bare])


# --- Each member is a Chain ---------------------------------------------------


def _standalone(evolver, seed, init_fields, **kwargs):
    chain = Chain(evolver=evolver, seed=seed, init_fields=init_fields, **kwargs)
    chain.warmup(4)
    chain.run(6)
    return chain


def test_each_member_is_the_chain_it_would_be_alone(evolvers, lat4):
    # The central guarantee.  Members differ in kappa, eps, seed and start.
    starts = [{'phi': _random_phi(lat4, i)} for i in range(B)]
    kwargs = dict(measure=[mag, (profile, 2)], save_freq=3)

    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields=starts, **kwargs)
    mc.warmup(4)
    mc.run(6)

    for i in range(B):
        alone = _standalone(evolvers[i], SEEDS[i], starts[i], **kwargs)

        for key in ('delta_H', 'P_acc'):
            assert np.allclose(mc.monitor[key][i], alone.monitor[key])
        assert np.array_equal(mc.monitor['accept'][i], alone.monitor['accept'])
        assert np.allclose(mc.column('m')[i], alone.column('m'))
        assert np.allclose(mc.column('C')[i], alone.column('C'))
        assert np.allclose(mc.fields['phi'].F[i], alone.fields['phi'].F)
        for saved, saved_alone in zip(mc.field_chain['phi'], alone.field_chain['phi']):
            assert np.allclose(saved.F[i], saved_alone.F)


def test_multichain_warns_on_duplicate_seeds(phi4):
    # Even across different parameters, reusing a seed is almost always a
    # mistake: it is allowed, but never silent.
    evolvers = [_hmc(kappa=k) for k in KAPPAS]
    with pytest.warns(UserWarning, match=r"seed.*\b5\b"):
        MultiChain(evolvers=evolvers, seeds=(5, 6, 5), init_fields={'phi': phi4})


def test_member_fields_are_ordinary_fields(evolvers, lat4):
    starts = [{'phi': _random_phi(lat4, i)} for i in range(B)]
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields=starts)
    mc.run(3)

    for i in range(B):
        alone = Chain(evolver=evolvers[i], seed=SEEDS[i], init_fields=starts[i])
        alone.run(3)

        phi = mc.member_fields(i)['phi']
        assert isinstance(phi, lat.LatticeField)
        assert phi.lattice == lat4
        assert phi.F.shape == (4, 4, 4)
        assert np.allclose(phi.F, alone.fields['phi'].F)
        assert np.allclose(mag({'phi': phi}, None)['m'], mag(alone.fields, None)['m'])


def test_member_fields_takes_python_indices(evolvers, phi4):
    # Negative indices count from the end; an index past the end raises,
    # where indexing a JAX array would silently clamp it to the last member.
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4})
    mc.run(2)

    assert np.array_equal(mc.member_fields(-1)['phi'].F, mc.member_fields(B - 1)['phi'].F)
    with pytest.raises(IndexError):
        mc.member_fields(B)
    with pytest.raises(IndexError):
        mc.member_fields(-B - 1)


def test_members_with_one_seed_are_identical(phi4):
    # The batch shares no randomness between members: the seed alone decides.
    with pytest.warns(UserWarning, match="seed"):
        mc = MultiChain(evolvers=[_hmc()] * 3, seeds=(5, 6, 5), init_fields={'phi': phi4})
    mc.run(4)

    dH = np.asarray(mc.monitor['delta_H'])
    assert np.array_equal(dH[0], dH[2])
    assert not np.array_equal(dH[0], dH[1])


def test_each_member_measures_with_its_own_action(evolvers, phi4):
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, measure=[coupling])
    mc.run(2)

    assert mc.column('kappa').shape == (B, 2)
    assert np.allclose(mc.column('kappa'), np.array(KAPPAS)[:, None])


# --- Shapes and bookkeeping ---------------------------------------------------


def test_multichain_outputs_lead_with_the_batch_axis(evolvers, phi4):
    mc = MultiChain(
        evolvers=evolvers,
        seeds=SEEDS,
        init_fields={'phi': phi4},
        measure=[mag, (profile, 2)],
        save_freq=2,
    )
    mc.run(6)

    assert np.shape(mc.monitor['delta_H']) == (B, 6)
    assert mc.column('m').shape == (B, 6)
    assert mc.column('C').shape == (B, 3, 4)
    assert mc.column('C').dtype == np.float64
    assert list(mc.column_traj('C')) == [2, 4, 6]
    assert mc.traj_chain == [0, 2, 4, 6]
    assert all(f.F.shape == (B, 4, 4, 4) for f in mc.field_chain['phi'])


def test_multichain_columns_align_on_shared_trajectories(evolvers, phi4):
    mc = MultiChain(
        evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, measure=[mag, (profile, 3)]
    )
    mc.run(6)

    m, C = mc.columns('m', 'C')
    assert m.shape == (B, 2)
    assert C.shape == (B, 2, 4)
    assert np.array_equal(m, mc.column('m')[:, [2, 5]])


def test_multichain_empty_column_keeps_its_shape(evolvers, phi4):
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, measure=[profile])

    assert mc.column('C').shape == (B, 0, 4)


def test_multichain_warmup_saves_and_measures_nothing(evolvers, phi4):
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, measure=[mag])
    mc.warmup(5)

    assert mc.warmup_traj == 5
    assert mc.traj == 0
    assert mc.traj_chain == [0]
    assert mc.column('m').shape == (B, 0)
    assert np.shape(mc.monitor['accept']) == (B, 5)
    assert np.all(mc.monitor['accept'])


def test_multichain_acceptance_is_per_member_and_production_only(evolvers, phi4):
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4})
    assert mc.acceptance() is None

    mc.warmup(5)
    mc.run(10)

    acc = mc.acceptance()
    assert acc.shape == (B,)
    assert np.allclose(acc, np.mean(np.asarray(mc.monitor['accept'])[:, -10:], axis=1))


def test_multichain_run_requires_commensurate_ntraj(evolvers, phi4):
    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, save_freq=4)
    with pytest.raises(ValueError, match="must be a multiple of 4"):
        mc.run(6)


def test_multichain_measure_is_compiled_once(evolvers, phi4):
    traces = []

    def counted(fields, action):
        traces.append(1)
        return {'m': jnp.mean(fields['phi'].F)}

    mc = MultiChain(evolvers=evolvers, seeds=SEEDS, init_fields={'phi': phi4}, measure=[counted])
    mc.run(3)
    n = len(traces)
    mc.run(5)

    assert len(traces) == n
    assert mc.column('m').shape == (B, 8)


# --- Solves -------------------------------------------------------------------

L2 = lat.SquareLattice(st_dims=(6, 6))

# A fixed random potential spreads the spectrum.  The free Laplacian on a small
# lattice has so few distinct eigenvalues that CG converges exactly in the same
# few iterations for any m2.
V = lat.LatticeField(lattice=L2, F=jax.random.uniform(jax.random.PRNGKey(1), (6, 6)))


def _operator(m2):
    def A(f):
        hops = f.nn_field(0, 1) + f.nn_field(0, -1) + f.nn_field(1, 1) + f.nn_field(1, -1)
        return (m2 + 4) * f + V * f - hops

    return A


class Gaussian(Term):
    """S = 1/2 phi . A^-1 phi by a CG solve; `tol` is traced, so it may vary."""

    m2: float
    tol: float = 1e-10
    maxiter: int = eqx.field(static=True, default=1000)

    def total(self, phi):
        x, info = cg(_operator(self.m2), phi, tol=self.tol, maxiter=self.maxiter)
        return 0.5 * jnp.vdot(phi.F, x.F).real, info


def _gaussian_hmc(**kwargs):
    return HMC(action=Gaussian(**kwargs), integrator=LeapfrogIntegrator(eps=0.1, Nstep=3))


def _phi2():
    return {'phi': lat.LatticeField(lattice=L2, F=jax.random.normal(jax.random.PRNGKey(0), (6, 6)))}


def test_solver_iterations_are_counted_per_member():
    # Under vmap a CG loop runs until the slowest member converges, but each
    # member's own count must still be its own.
    evolvers = [_gaussian_hmc(m2=m2) for m2 in (0.05, 2.0)]
    mc = MultiChain(evolvers=evolvers, seeds=(1, 2), init_fields=_phi2())
    mc.run(2)

    iters = np.asarray(mc.monitor['Gaussian.iterations'])
    assert iters.shape == (2, 2, 5)  # (member, traj, solves per traj)
    assert iters[0].mean() > iters[1].mean()  # the light mass is the slow one
    for i in range(2):
        alone = Chain(evolver=evolvers[i], seed=(1, 2)[i], init_fields=_phi2())
        alone.run(2)
        assert np.array_equal(iters[i], alone.monitor['Gaussian.iterations'])


def test_multichain_warns_naming_the_members_that_did_not_converge():
    # Member 1 asks for more accuracy than 5 iterations can give.
    evolvers = [_gaussian_hmc(m2=1.0, tol=1e-1, maxiter=5), _gaussian_hmc(m2=1.0, maxiter=5)]
    mc = MultiChain(evolvers=evolvers, seeds=(1, 2), init_fields=_phi2())

    with pytest.warns(RuntimeWarning, match=r"converge.*\[1\]"):
        mc.run(1)


def test_multichain_is_quiet_when_solves_converge():
    evolvers = [_gaussian_hmc(m2=1.0), _gaussian_hmc(m2=2.0)]
    mc = MultiChain(evolvers=evolvers, seeds=(1, 2), init_fields=_phi2())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mc.warmup(1).run(1)
