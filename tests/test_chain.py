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


def cheap(fields, action):
    return {'m': jnp.mean(fields['phi'].F)}


def expensive(fields, action):
    phi = fields['phi'].F
    return {'m2': jnp.mean(phi**2), 'C': jnp.mean(phi, axis=(1, 2))}


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


def test_chain_measure_respects_frequency(hmc, phi4):
    chain = Chain(
        evolver=hmc,
        seed=1,
        init_fields={'phi': phi4},
        measure=[(cheap, 4)],
        save_freq=2,
    )
    chain.run(12)

    assert list(chain.column_traj('m')) == [4, 8, 12]
    assert chain.column('m').shape == (3,)
    assert np.all(np.isfinite(chain.column('m')))


def test_chain_bare_measure_function_runs_every_traj(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[cheap])
    chain.run(5)

    assert list(chain.column_traj('m')) == [1, 2, 3, 4, 5]


def test_chain_measures_the_current_configuration(hmc, phi4):
    # Each value must come from the configuration at its own trajectory.
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[(cheap, 2)])
    chain.run(6)

    saved = {t: f for t, f in zip(chain.traj_chain, chain.field_chain['phi'])}
    expected = [float(jnp.mean(saved[t].F)) for t in chain.column_traj('m')]
    assert np.allclose(chain.column('m'), expected)


def test_chain_multiple_measure_frequencies(hmc, phi4):
    chain = Chain(
        evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[cheap, (expensive, 3)]
    )
    chain.run(6)

    assert list(chain.column_traj('m')) == [1, 2, 3, 4, 5, 6]
    assert list(chain.column_traj('m2')) == [3, 6]
    assert list(chain.column_traj('C')) == [3, 6]


def test_chain_column_is_float64_with_measurement_shape(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[expensive])
    chain.run(5)

    C = chain.column('C')
    assert C.dtype == np.float64
    assert C.shape == (5, 4)  # (Monte Carlo time, t)


def test_chain_columns_aligns_on_shared_trajectories(hmc, phi4):
    chain = Chain(
        evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[cheap, (expensive, 3)]
    )
    chain.run(9)

    m, m2 = chain.columns('m', 'm2')
    assert m.shape == m2.shape == (3,)
    # 'm' is measured every trajectory; only trajectories 3, 6, 9 are shared.
    assert np.array_equal(m, chain.column('m')[[2, 5, 8]])
    assert np.array_equal(m2, chain.column('m2'))


def test_chain_empty_column_keeps_measurement_shape(hmc, phi4):
    # Before any measurement, a vector column is (0, T), not (0,), so code
    # that indexes its trailing axis does not need a special case.
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[expensive])

    assert chain.column('C').shape == (0, 4)
    assert chain.column('m2').shape == (0,)


def test_chain_column_unknown_name(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[cheap])
    with pytest.raises(KeyError, match="'nope'"):
        chain.column('nope')


def test_chain_measure_receives_the_action(hmc, phi4, scalar_params):
    def coupling(fields, action):
        return {'kappa': action['ScalarTerm'].kappa}

    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[coupling])
    chain.run(3)

    assert np.allclose(chain.column('kappa'), scalar_params['kappa'])


def test_chain_rejects_colliding_measurement_names(hmc, phi4):
    # Caught at construction, before any trajectory is run.
    def also_m(fields, action):
        return {'m': jnp.max(fields['phi'].F)}

    with pytest.raises(ValueError, match="'m'"):
        Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[cheap, also_m])


def test_chain_rejects_measure_not_returning_a_dict(hmc, phi4):
    def bare(fields, action):
        return jnp.mean(fields['phi'].F)

    with pytest.raises(TypeError, match="dict"):
        Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[bare])


def test_chain_block_size_includes_measure_frequencies(hmc, phi4):
    chain = Chain(
        evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[(cheap, 6)], save_freq=4
    )
    with pytest.raises(ValueError, match="must be a multiple of 2"):
        chain.run(3)
    chain.run(12)
    assert list(chain.column_traj('m')) == [6, 12]


def test_chain_measure_is_compiled_once(hmc, phi4):
    # Chain compiles measurement functions itself: the Python body runs at
    # trace time only, however many trajectories are measured.
    traces = []

    def counted(fields, action):
        traces.append(1)
        return {'m': jnp.mean(fields['phi'].F)}

    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[counted])
    chain.run(3)
    n = len(traces)
    chain.run(5)

    assert len(traces) == n
    assert len(chain.column('m')) == 8


def test_chain_warmup_does_not_measure(hmc, phi4):
    chain = Chain(evolver=hmc, seed=1, init_fields={'phi': phi4}, measure=[cheap])
    chain.warmup(5)

    assert len(chain.column('m')) == 0


def test_chain_measuring_does_not_change_the_markov_chain(hmc, phi4):
    # Measurement frequency changes the block size, which must not change
    # the chain itself.
    plain = Chain(evolver=hmc, seed=3, init_fields={'phi': phi4}, save_freq=8)
    measured = Chain(
        evolver=hmc, seed=3, init_fields={'phi': phi4}, save_freq=8, measure=[(cheap, 2)]
    )
    plain.run(8)
    measured.run(8)

    assert np.allclose(plain.fields['phi'].F, measured.fields['phi'].F)


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
