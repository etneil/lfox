import jax
import jax.numpy as jnp
import numpy as np
import lfox.lattice as lat
from lfox.evolution import HMC, LeapfrogIntegrator, OmelyanIntegrator
from tests.phi4 import ScalarAction


def test_action_uniform_field(scalar_action, lat4, scalar_params):
    # For uniform field c on a d-dim lattice with periodic BC:
    # S per site = c^2*(1 - 2*kappa*d) + lambda*(c^2-1)^2
    c = 1.5
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), c))
    kappa, lam, d = scalar_params['kappa'], scalar_params['lambda'], 3
    expected = 4**3 * (c**2 * (1 - 2*kappa*d) + lam * (c**2 - 1)**2)
    assert jnp.isclose(scalar_action.S({'phi': phi}), expected, rtol=1e-5)


def test_action_field_density_shape(scalar_action, lat4, phi4):
    S_density = scalar_action.S_field({'phi': phi4})
    assert S_density.F.shape == (4, 4, 4)
    assert jnp.all(jnp.isfinite(S_density.F))


def test_force_matches_exact(scalar_action, lat4, scalar_params):
    # Autodiff dS should agree with the analytic force on a random field
    rng = jax.random.PRNGKey(0)
    F_vals = jax.random.normal(rng, (4, 4, 4), dtype=jnp.float64)
    phi = lat.LatticeField(lattice=lat4, F=F_vals)

    autodiff_force = scalar_action.dS({'phi': phi})['phi'].F
    exact_force = ScalarAction.exact_force([phi], scalar_params).F

    assert jnp.allclose(autodiff_force, exact_force, rtol=1e-5)


def test_hmc_single_step_monitor_keys(scalar_action, lat4, phi4):
    hmc = HMC(
        action=scalar_action,
        integrator=LeapfrogIntegrator(eps=0.1, Nstep=10),
    )
    _, monitor, _ = hmc.evolve({'phi': phi4}, jax.random.PRNGKey(42))
    assert {'delta_H', 'P_acc', 'accept'} <= monitor.keys()


def test_hmc_single_step_finite(scalar_action, lat4, phi4):
    hmc = HMC(
        action=scalar_action,
        integrator=LeapfrogIntegrator(eps=0.1, Nstep=10),
    )
    _, monitor, _ = hmc.evolve({'phi': phi4}, jax.random.PRNGKey(7))
    assert jnp.isfinite(monitor['delta_H'])
    assert jnp.isfinite(monitor['P_acc'])


def test_hmc_omelyan_finite(scalar_action, lat4, phi4):
    hmc = HMC(
        action=scalar_action,
        integrator=OmelyanIntegrator(eps=0.1, Nstep=10),
    )
    new_fields, monitor, _ = hmc.evolve({'phi': phi4}, jax.random.PRNGKey(99))
    assert jnp.isfinite(monitor['delta_H'])
    assert new_fields['phi'].F.shape == (4, 4, 4)


def test_hmc_evolve_advances_rng_key(scalar_action, phi4):
    # The accept/reject uniform must be drawn from a key advanced past every
    # momentum draw, otherwise it is a deterministic function of the momenta it
    # is supposed to be testing.  Pin that the returned key is neither the input
    # key nor the momentum subkey.
    hmc = HMC(
        action=scalar_action,
        integrator=LeapfrogIntegrator(eps=0.1, Nstep=5),
    )
    rng_key = jax.random.PRNGKey(11)
    _, _, new_key = hmc.evolve({'phi': phi4}, rng_key)

    mom_key, mom_subkey = jax.random.split(rng_key)
    assert not jnp.array_equal(new_key, rng_key)
    assert not jnp.array_equal(new_key, mom_subkey)
    # ...and it is advanced past the single-field momentum draw, not equal to it.
    assert not jnp.array_equal(new_key, mom_key)


def test_hmc_warmup_always_accepts(scalar_action, phi4):
    # A warmup kernel skips the accept/reject step entirely.
    hmc = HMC(
        action=scalar_action,
        integrator=LeapfrogIntegrator(eps=0.5, Nstep=5),  # coarse: forces rejections
    )
    _, monitor, _ = hmc.as_warmup().evolve_many(
        {'phi': phi4}, jax.random.PRNGKey(5), traj=20
    )
    assert jnp.all(monitor['accept'])

    # ...whereas the production kernel rejects sometimes at this step size.
    _, monitor, _ = hmc.evolve_many({'phi': phi4}, jax.random.PRNGKey(5), traj=20)
    assert not jnp.all(monitor['accept'])


def test_hmc_monitor_is_stacked_per_trajectory(scalar_action, phi4):
    # evolve_many stacks whatever monitor the kernel returns, with dtypes intact.
    hmc = HMC(
        action=scalar_action,
        integrator=LeapfrogIntegrator(eps=0.1, Nstep=10),
    )
    _, monitor, _ = hmc.evolve_many({'phi': phi4}, jax.random.PRNGKey(3), traj=7)

    assert monitor['delta_H'].shape == (7,)
    assert monitor['accept'].shape == (7,)
    assert monitor['accept'].dtype == jnp.bool_


def test_hmc_detailed_balance():
    # Metropolis requires <exp(-delta_H)> = 1.
    # 500 trajectories on a 4^3 lattice with fixed seed is enough to verify this
    # isn't grossly violated (true value ≈ 1, tolerance set to ±10%).
    L = lat.SquareLattice(st_dims=(4, 4, 4))
    phi = lat.LatticeField(lattice=L, F=jnp.ones((4, 4, 4)))
    action = ScalarAction(
        field_names=['phi'], params={'kappa': 0.18, 'lambda': 1.145}
    )
    hmc = HMC(
        action=action,
        integrator=LeapfrogIntegrator(eps=0.1, Nstep=10),
    )

    fields = {'phi': phi}
    rng_key = jax.random.PRNGKey(12345)
    fields, _, rng_key = hmc.as_warmup().evolve_many(fields, rng_key, traj=100)
    fields, monitor, _ = hmc.evolve_many(fields, rng_key, traj=500)

    mean_exp = float(np.mean(np.exp(-np.array(monitor['delta_H']))))
    assert 0.9 < mean_exp < 1.1, f"<exp(-dH)> = {mean_exp:.4f}, expected ~1.0"
