import jax.numpy as jnp
import pytest

import lfox.lattice as lat


def test_square_lattice_dims():
    L = lat.SquareLattice(st_dims=(4, 4, 4))
    assert L.st_dims == (4, 4, 4)
    assert L._dims == (4, 4, 4)


def test_lattice_field_shape(lat4):
    phi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)))
    assert phi.F.shape == (4, 4, 4)
    assert phi.dims() == (4, 4, 4)
    assert phi.d() == 3


def test_lattice_field_constant_value(lat4):
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 2.5))
    assert jnp.allclose(phi.F, 2.5)


def test_lattice_field_default_bc(lat4):
    phi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)))
    assert phi.bc == (1, 1, 1)


def test_lattice_field_add(lat4):
    phi1 = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 2.0))
    phi2 = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 3.0))
    result = phi1 + phi2
    assert jnp.allclose(result.F, 5.0)


def test_lattice_field_scalar_mul(lat4):
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 2.0))
    assert jnp.allclose((phi * 3.0).F, 6.0)
    assert jnp.allclose((3.0 * phi).F, 6.0)


def test_lattice_field_pow(lat4):
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 3.0))
    assert jnp.allclose((phi**2).F, 9.0)


def test_lattice_field_neg(lat4):
    # A new field with the same boundary conditions; the original is unchanged.
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 2.0), bc=(1, 1, -1))
    neg = -phi
    assert jnp.allclose(neg.F, -2.0)
    assert neg.bc == phi.bc
    assert jnp.allclose(phi.F, 2.0)


def test_lattice_field_reflected_sub_and_div(lat4):
    # A scalar on the left, as in 1 - phi**2 or 1 / (1 + g * phi**2).
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 2.0))
    assert jnp.allclose((1.0 - phi).F, -1.0)
    assert jnp.allclose((6.0 / phi).F, 3.0)
    assert isinstance(1.0 - phi, lat.LatticeField)
    assert isinstance(6.0 / phi, lat.LatticeField)


def test_nn_field_uniform_periodic(lat4):
    # Uniform field with periodic BC: any shift returns the same values
    phi = lat.LatticeField(lattice=lat4, F=jnp.full((4, 4, 4), 3.7))
    for ax in range(3):
        assert jnp.allclose(phi.nn_field(axis=ax, shift=1).F, phi.F)
        assert jnp.allclose(phi.nn_field(axis=ax, shift=-1).F, phi.F)


def test_nn_field_periodic_roll():
    # roll(shift=1) gives each site the value from site i-1 (left neighbor wraps)
    L = lat.SquareLattice(st_dims=(4,))
    phi = lat.LatticeField(lattice=L, F=jnp.array([1.0, 2.0, 3.0, 4.0]))
    shifted = phi.nn_field(axis=0, shift=1)
    assert jnp.allclose(shifted.F, jnp.array([4.0, 1.0, 2.0, 3.0]))


def test_nn_field_periodic_roll_negative():
    # roll(shift=-1) gives each site the value from site i+1 (right neighbor wraps)
    L = lat.SquareLattice(st_dims=(4,))
    phi = lat.LatticeField(lattice=L, F=jnp.array([1.0, 2.0, 3.0, 4.0]))
    shifted = phi.nn_field(axis=0, shift=-1)
    assert jnp.allclose(shifted.F, jnp.array([2.0, 3.0, 4.0, 1.0]))


def test_nn_field_antiperiodic_boundary():
    # roll(shift=1) gives result[i] = F[(i-1) % N]: the left neighbor.
    # Site 0 is the one that wraps (reads from site N-1 across the boundary),
    # so anti-periodic BC should flip the sign only at site 0.
    # Correct expected: [-4, 1, 2, 3]
    L = lat.SquareLattice(st_dims=(4,))
    phi = lat.LatticeField(lattice=L, F=jnp.array([1.0, 2.0, 3.0, 4.0]), bc=(-1,))
    shifted = phi.nn_field(axis=0, shift=1)
    assert jnp.allclose(shifted.F, jnp.array([-4.0, 1.0, 2.0, 3.0]))

    # Shift it again to make sure we get: [-3, -4, 1, 2]
    shifted_twice = shifted.nn_field(axis=0, shift=1)
    assert jnp.allclose(shifted_twice.F, jnp.array([-3.0, -4.0, 1.0, 2.0]))

    # Shift it around fully to make sure we get the same field back with a sign
    shifted_L = phi.nn_field(axis=0, shift=4)
    assert jnp.allclose(shifted_L.F, (-1) * phi.F)

    # Shift around twice to make sure we get the same field back identically
    shifted_2L = phi.nn_field(axis=0, shift=8)
    assert jnp.allclose(shifted_2L.F, phi.F)


def test_lattice_volume():
    assert lat.SquareLattice(st_dims=(6, 4, 8)).volume == 192
    # Counts sites, so a unit cell multiplies it.
    assert lat.HoneycombLattice(st_dims=(3, 4)).volume == 24


# --- the shape of F ---------------------------------------------------------
#
# KNOWN FAILURES until LatticeField checks F.  Its __post_init__ multiplies F
# by ones(dims()) -- meant for a scalar, but done for every F -- so a wrongly
# shaped array is silently broadcast against the lattice instead of rejected.


@pytest.mark.parametrize("shape", [(4,), (4, 4), (4, 1, 4), (1, 4, 4, 4), (2, 4, 4, 4)])
def test_lattice_field_rejects_a_wrongly_shaped_array(lat4, shape):
    with pytest.raises(ValueError, match="has shape"):
        lat.LatticeField(lattice=lat4, F=jnp.ones(shape))


def test_lattice_field_shape_includes_internal_indices(lat4):
    # A field with indices=(2,) has shape dims + (2,); a spacetime-only F is
    # rejected rather than broadcast across the index.
    psi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4, 2)), indices=(2,))
    assert psi.F.shape == (4, 4, 4, 2)

    with pytest.raises(ValueError, match="has shape"):
        lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)), indices=(2,))


def test_lattice_field_arithmetic_cannot_change_its_shape(lat4):
    # Broadcasting in an operator must not quietly turn a field into a batch
    # of fields.
    phi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)))
    with pytest.raises(ValueError, match="has shape"):
        phi * jnp.ones((2, 4, 4, 4))


def test_lattice_field_from_a_scalar_is_a_float_field(lat4):
    # F=1 is the shorthand for a unit field.  An integer F -- scalar or array
    # -- still gives a float field, or autodiff (dS) would refuse it.
    phi = lat.LatticeField(lattice=lat4, F=1)
    assert phi.F.shape == (4, 4, 4)
    assert jnp.issubdtype(phi.F.dtype, jnp.floating)
    assert jnp.all(phi.F == 1)

    ints = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4), dtype=int))
    assert jnp.issubdtype(ints.F.dtype, jnp.floating)
