import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import lfox.lattice as lat
from lfox.action import Action
from tests.phi4 import ScalarTerm  # noqa: F401 — re-exported for convenience


@pytest.fixture
def scalar_params():
    return {'kappa': 0.18169, 'lamb': 1.3282}


@pytest.fixture
def scalar_action(scalar_params):
    return Action(ScalarTerm(**scalar_params))


@pytest.fixture
def lat4():
    return lat.SquareLattice(st_dims=(4, 4, 4))


@pytest.fixture
def phi4(lat4):
    return lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)))
