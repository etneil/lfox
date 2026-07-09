import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.evolution import HMC, Chain, LeapfrogIntegrator


@pytest.fixture
def hmc(scalar_action):
    return HMC(
        action=scalar_action,
        integrator=LeapfrogIntegrator(eps=0.1, Nstep=10),
    )


def magnetization(fields, params):
    return float(jnp.mean(fields['phi'].F))


def test_chain_records_initial_config(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4})
    assert chain.traj == 0
    assert chain.traj_chain == [0]
    assert len(chain.field_chain['phi']) == 1


def test_chain_rejects_mismatched_field_names(hmc, phi4):
    with pytest.raises(ValueError, match="do not match"):
        Chain(evolver=hmc, seed=1, init_fields={'psi': phi4})


def test_chain_run_saves_every_trajectory(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4})
    chain.run(6)

    assert chain.traj == 6
    assert chain.traj_chain == [0, 1, 2, 3, 4, 5, 6]
    assert len(chain.field_chain['phi']) == 7
    assert len(chain.monitor['delta_H']) == 6


def test_chain_save_freq(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, save_freq=3)
    chain.run(9)

    # Configs saved every 3 trajectories; diagnostics recorded every trajectory.
    assert chain.traj_chain == [0, 3, 6, 9]
    assert len(chain.field_chain['phi']) == 4
    assert len(chain.monitor['delta_H']) == 9


def test_chain_run_requires_commensurate_ntraj(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, save_freq=4)
    with pytest.raises(ValueError, match="must be a multiple of 4"):
        chain.run(6)


def test_chain_observables_respect_frequency(hmc, phi4):
    chain = Chain(
        evolver=hmc,
        seed=1,
        init_fields={'phi': phi4},
        observables={'magn': [magnetization, 4]},
        save_freq=2,
    )
    chain.run(12)

    assert chain.obs_traj['magn'] == [4, 8, 12]
    assert len(chain.obs_chain['magn']) == 3
    assert all(np.isfinite(chain.obs_chain['magn']))


def test_chain_bare_callable_observable_measures_every_traj(hmc, phi4):
    chain = Chain(
        evolver=hmc, seed=1, init_fields={'phi': phi4}, observables={'magn': magnetization}
    )
    chain.run(5)

    assert chain.obs_traj['magn'] == [1, 2, 3, 4, 5]


def test_chain_warmup_does_not_save_configs(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4})
    chain.warmup(10)

    assert chain.warmup_traj == 10
    assert chain.traj == 0
    # Warmup configurations are not samples of the target distribution.
    assert chain.traj_chain == [0]
    assert len(chain.field_chain['phi']) == 1
    # ...but diagnostics from warmup are still recorded.
    assert len(chain.monitor['delta_H']) == 10
    assert all(chain.monitor['accept'])


def test_chain_acceptance_uses_production_only(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4})
    assert chain.acceptance() is None  # nothing run yet

    chain.warmup(10)  # always accepts; must not inflate the production estimate
    chain.run(20)

    acc = chain.acceptance()
    assert acc == pytest.approx(np.mean(chain.monitor['accept'][-20:]))
    assert 0.0 <= acc <= 1.0


def test_chain_is_reproducible_from_seed(hmc, phi4):
    def final_field(seed):
        chain = Chain(evolver=hmc, seed=seed, init_fields={'phi': phi4})
        chain.warmup(5)
        chain.run(5)
        return np.array(chain.fields['phi'].F)

    assert np.array_equal(final_field(7), final_field(7))
    assert not np.array_equal(final_field(7), final_field(8))


def test_chain_blocking_does_not_change_the_markov_chain(scalar_action, phi4):
    # Stepping in blocks of save_freq must produce the same chain as stepping one
    # trajectory at a time -- the block size is a performance knob, not physics.
    hmc = HMC(action=scalar_action, integrator=LeapfrogIntegrator(eps=0.1, Nstep=10))

    fine = Chain(evolver=hmc, seed=3, init_fields={'phi': phi4}, save_freq=1)
    coarse = Chain(evolver=hmc, seed=3, init_fields={'phi': phi4}, save_freq=4)
    fine.run(8)
    coarse.run(8)

    assert np.allclose(fine.fields['phi'].F, coarse.fields['phi'].F)
    assert np.allclose(fine.monitor['delta_H'], coarse.monitor['delta_H'])
