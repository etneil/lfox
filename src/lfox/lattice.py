import copy
import dataclasses
import operator
from abc import ABC, abstractmethod
from functools import partial
from typing import ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util


class Lattice(eqx.Module):
    st_dims: tuple[int]
    _dims: tuple[int] = eqx.field(init=False)
    _bc_coords: jax.Array = eqx.field(init=False)
    #    st_dims: jax.Array = eqx.field(converter=jax.numpy.asarray)
    #    _dims: jax.Array = eqx.field(init=False)

    def __post_init__(self):
        # Dimensions of the physical lattice
        # May differ from space-time dims (e.g.
        # non-trivial unit cell)
        self._dims = self.st_dims

        # Field used for application of boundary conditions in LatticeFields
        # self._bc_coords = tuple(jnp.meshgrid(*[jnp.arange(Li) for Li in self.st_dims], indexing='ij'))
        self._bc_coords = tuple(
            jnp.meshgrid(*[jnp.arange(Li) for Li in self.st_dims], indexing="ij")
        )

    # Override equality since it's using the bc_coords field when it shouldn't be
    def __eq__(self, other):
        if not isinstance(other, Lattice):
            raise NotImplementedError

        return self.st_dims == other.st_dims

    def __hash__(self):
        return hash(self.st_dims)


class SquareLattice(Lattice):
    def shift(self, field, axis, shift=1):
        return jnp.roll(field, shift=shift, axis=axis)

    # Red-black checkerboarding
    # rb fields are defined on a sublattice with 1/2 size.
    def rb_split(self, field):
        # The goal is to figure this out efficiently, but
        # it seems very tricky especially with boundary conditions...
        # Return to it later.

        pass
        # return cb_red, cb_black

    def rb_combine(self, field_cb_red, field_cb_black):
        # Inverse of rb_split, however that ends up being implemented!
        pass
        # return field

    @staticmethod
    @jax.jit
    def _set_checkerboard(dims):
        cb_black = jnp.indices(dims).sum(axis=0) % 2
        cb_red = (cb_black + 1) % 2

        return [cb_black, cb_red]


class HoneycombLattice(Lattice):
    # A/B unit cell structure, automatically in the dim-0 direction.
    # Note that this means that the number of lattice sites in dimension
    # 0 is actually 2*dims[0].
    unit_cell: ClassVar[tuple[int]] = (0, 1)

    def __post_init__(self):
        #        self._dims = jnp.concatenate((self.st_dims, jnp.asarray((len(self.unit_cell),))))
        self._dims = self.st_dims + (len(self.unit_cell),)

        # Field used for application of boundary conditions in LatticeFields
        self._bc_coords = jnp.meshgrid(
            *[jnp.arange(Li) for Li in self.st_dims], indexing="ij"
        )

    def shift(self, field, axis, shift=1):
        # Shift within the unit cell too if we are moving in axis 0
        if axis == 0:
            shift_field = jnp.roll(field, shift=shift, axis=-1)
            # FIXME: THIS IS WRONG, figure out the correct way to do this
            # (Only sites which are now on unit cell position 0 should be
            # rolled in x.)
            shift_field = jnp.roll(shift_field, shift=shift, axis=0)
            return shift_field
        else:
            # Other axes shift normally
            return jnp.roll(field, shift=shift, axis=axis)


class LatticeField(eqx.Module):
    lattice: Lattice = eqx.field(static=True)
    F: jax.Array = eqx.field(converter=jax.numpy.asarray)
    #    bc: Optional[jax.Array] = eqx.field(default=(), converter=jax.numpy.asarray)
    bc: Optional[tuple[int]] = eqx.field(default=(), static=True)
    indices: Optional[tuple[int]] = eqx.field(default=(), static=True)

    def __post_init__(self):
        # Allow e.g. unit field by just passing 1,
        # broadcast to size of lattice

        if type(self.F) != jnp.array:
            #        if self.F.shape == ():
            self.F = self.F * jnp.ones(self.dims())

        self._set_default_BC()

        if type(self.lattice) is dict:
            # Workaround for using dataclasses.asdict
            self.lattice = Lattice(**self.lattice)

    def _set_default_BC(self):
        if len(self.bc) == 0:
            # Default is periodic BC = [1,1,1,...]
            self.bc = (1,) * len(self.lattice.st_dims)
        #            self.bc = jnp.ones(len(self.lattice.st_dims))
        else:
            assert len(self.bc) == len(self.lattice.st_dims)

    def dims(self):
        if len(self.indices) > 0:
            return self.lattice._dims + tuple(self.indices)
        else:
            return self.lattice._dims

    def st_dims(self):
        # For broadcasting against spacetime indices
        return self.lattice._dims + (1,) * len(self.indices)

    def d(self):
        # Return number of space-time dimensions
        return len(self.lattice._dims)

    @staticmethod
    def _field_op(f, rev=False):
        if rev:

            def op(self, other):
                o = getattr(other, "F", other)
                new_F = f(o, self.F)

                return self.copy_new_F(new_F)
        else:

            def op(self, other):
                o = getattr(other, "F", other)
                new_F = f(self.F, o)

                return self.copy_new_F(new_F)

        return op

    __add__ = _field_op(operator.add)
    __radd__ = _field_op(operator.add, rev=True)
    __sub__ = _field_op(operator.sub)
    __mul__ = _field_op(operator.mul)
    __rmul__ = _field_op(operator.mul, rev=True)
    __truediv__ = _field_op(operator.truediv)
    __pow__ = _field_op(operator.pow)

    def conj(self):
        new_F = self.F.conj()
        return self.copy_new_F(new_F)

    # TODO: more arithmetic?

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
    def _nn_field(field, lattice, axis, shift, bc, st_dims):
        nn_shift = lattice.shift(field, axis=axis, shift=shift)

        # Apply boundary conditions globally with some arcane NumPy manipulations
        BC_factor = bc[axis]
        #        coords_ax = jnp.meshgrid(*[jnp.arange(Li) for Li in st_dims], indexing='ij')[axis]
        coords_ax = lattice._bc_coords[axis]

        # Minus signs show up at target, not source, so subtract shift.
        winding, _ = jnp.divmod(coords_ax - shift, st_dims[axis])
        BC_field = (BC_factor) ** (winding)

        return nn_shift * BC_field.reshape(st_dims)

    def nn_field(self, axis, shift=1):
        shifted_field = self._nn_field(
            self.F,
            lattice=self.lattice,
            axis=axis,
            shift=shift,
            bc=self.bc,
            st_dims=self.st_dims(),
        )

        return self.copy_new_F(shifted_field)

    """
    @partial(jax.jit, static_argnums=(1,))
    def nn_field(self, axis, shift=1):
        nn_shift = self.lattice.shift(self.F, axis=axis, shift=shift)

        # Apply boundary conditions globally with some arcane NumPy manipulations
        BC_factor = self.bc[axis]  # 1 or -1

        # Unpack for JIT compilation
        def grid_base(i, cgrid):
            Li = self.lattice.st_dims[i]
            return cgrid + [jnp.arange(Li)]

        cgrid = jax.lax.fori_loop(0, len(self.lattice.st_dims), grid_base, [])
        coords_ax = jnp.meshgrid(*cgrid, indexing='ij')[axis]

#        coords_ax = jnp.meshgrid(*[jnp.arange(Li) for Li in self.lattice.st_dims], indexing='ij')[axis]

        _, winding = jnp.divmod(coords_ax+shift, self.lattice.st_dims[axis])
        BC_field = (BC_factor)**(winding)

        LF = self.copy()
        LF.F = nn_shift * BC_field.reshape(self.st_dims())

        return LF
    """

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)

        return new

    def copy_new_F(self, F):
        # Unsure if this is the best way to do this?
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)

        return dataclasses.replace(new, F=F)

    def copy(self):
        return self.__copy__()
