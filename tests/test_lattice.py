import jax.numpy as jnp
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
    phi = lat.LatticeField(
        lattice=L, F=jnp.array([1.0, 2.0, 3.0, 4.0]), bc=(-1,)
    )
    shifted = phi.nn_field(axis=0, shift=1)
    assert jnp.allclose(shifted.F, jnp.array([-4.0, 1.0, 2.0, 3.0]))
