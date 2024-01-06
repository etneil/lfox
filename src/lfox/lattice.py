import numpy as np
from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from jax import tree_util
from functools import partial
from typing import ClassVar, Optional
import copy
import equinox as eqx
import dataclasses
import operator

class OldLattice(ABC):
    
    def __init__(self, dims):

        # self.dims contains the spacetime indices;
        # self._dims includes any internal unit-cell indices.
        # TODO: find a better/less confusing way to represent this...?

        self.dims = dims
        self._dims = dims
        self.d = len(dims)

    # "Shift" function, that will take an arbitrary
    # array with dimensions matching self.dims and shift it appropriately.
    # Boundary conditions to be applied within LatticeField objects
    # (since two fields on the same lattice can have different BCs.)
    @abstractmethod
    def shift(self, field, axis, shift=1):
        pass

    def _tree_flatten(self):
        children = (self.dims,)
        aux_data = {}

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(dims=children[0])

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)

        return new

tree_util.register_pytree_node(
    OldLattice,
    OldLattice._tree_flatten,
    OldLattice._tree_unflatten,
)

class Lattice(eqx.Module):
    st_dims: tuple[int]
    _dims: tuple[int] = eqx.field(init=False)

    def __post_init__(self):
        # Dimensions of the physical lattice
        # May differ from space-time dims (e.g.
        # non-trivial unit cell)
        self._dims = self.st_dims


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
        #return cb_red, cb_black
    
    def rb_combine(self, field_cb_red, field_cb_black):
        # Inverse of rb_split, however that ends up being implemented!
        pass
        #return field

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
        self._dims = self.st_dims + (len(self.unit_cell),)

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
    lattice: Lattice
    F: jax.Array = eqx.field(converter=jax.numpy.asarray)
    bc: Optional[jax.Array] = eqx.field(default=(), converter=jax.numpy.asarray)
    indices: Optional[tuple[int]] = ()

    def __post_init__(self):
        # Allow e.g. unit field by just passing 1,
        # broadcast to size of lattice
        if self.F.shape == ():
            self.F = self.F * jnp.ones(self.dims())

        self._set_default_BC()

        if type(self.lattice) is dict:
            # Workaround for using dataclasses.asdict
            self.lattice = Lattice(**self.lattice)

    def _set_default_BC(self):
        if len(self.bc) == 0:
            # Default is periodic BC = [1,1,1,...]
            self.bc = jnp.ones(len(self.lattice.st_dims))
        else:
            assert len(self.bc) == len(self.lattice.st_dims)
#            for B in self.bc:
                # Only (anti-)periodic BC supported for now
#                assert B in (-1, 1)

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
    def _field_op(f):
        def op(self, other):
            o = getattr(other, 'F', other)
            new_F = f(self.F, o)

            return self.copy_new_F(new_F)
        
        return op

    __add__ = _field_op(operator.add)
    __sub__ = _field_op(operator.sub)
    __mul__ = _field_op(operator.mul)
    __rmul__ = _field_op(operator.mul)
    __truediv__ = _field_op(operator.truediv)
    __pow__ = _field_op(operator.pow)
    
    def conj(self):
        new_F = self.F.conj()
        return self.copy_new_F(new_F)

    # TODO: more arithmetic?

    # JIT compiling this didn't seem useful in initial tests, at least as written...
    #@partial(jax.jit, static_argnums=(1,))
    @jax.jit
    def nn_field(self, axis, shift=1):
        nn_shift = self.lattice.shift(self.F, axis=axis, shift=shift)

        # Apply boundary conditions globally with some arcane NumPy manipulations
        BC_factor = self.bc[axis]  # 1 or -1
        coords_ax = jnp.meshgrid(*[jnp.arange(Li) for Li in self.lattice.dims], indexing='ij')[axis]

        _, winding = jnp.divmod(coords_ax+shift, self.lattice.dims[axis])
        BC_field = (BC_factor)**(winding)

        LF = self.copy()
        LF.F = nn_shift * BC_field.reshape(self.st_dims)

        return LF
    
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


class OldLatticeField:
    """
    Additional indices are represented as extra array dims in F
    that occur after the lattice dims.

    'indices' should be a sequence of integers giving the size of
    each of the extra dimensions of the field.  For example, given
    a Lattice with dims (6,6,6), and an indices tuple of (4,3) - say,
    a Dirac spinor with an SU(3) color index - the dimension of the 
    resulting field would be (6,6,6,4,3).
    """

    def __init__(self, lattice: Lattice, F=None, bc=None, indices=None):
        self.lattice = lattice
        self.dims = self.lattice._dims
        self.d = self.lattice.d

        if indices is not None:
            self.indices = indices
            self.dims += tuple(indices)
        else:
            self.indices = ()

        # Set up dimensions for spacetime broadcasting
        self.st_dims = self.lattice._dims
        if indices is not None:
            self.st_dims += (1,) * len(self.indices)

        if bc is None:
            # Default is periodic BC = [1,1,1,...]
            self.bc = np.ones(lattice.d)
        else:
            assert len(bc) == lattice.d
            for B in bc:
                # Only periodic and AP BC supported for now
                assert B in (-1, 1)
            
            self.bc = bc

        # Initialize the field
        if F is None:
            self._set_default_field()
        else:
            self.F = F

    def _set_default_field(self):
        self.F = jnp.zeros(self.dims)

    def _tree_flatten(self):
        children = (self.F,)
        aux_data = {
            'lattice': self.lattice,
            'bc': self.bc,
            'indices': self.indices,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(lattice=aux_data['lattice'], F=children[0], bc=aux_data['bc'], indices=aux_data['indices'])

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        # JAX/NumPy arrays are immutable, and this was slow for some reason
        # new.field = jnp.copy(self.field)
        new.F = self.F  

        return new
    
    def copy(self):
        return self.__copy__()


    # Arithmetic with fields - pass through to the field array
    def __add__(self, other):
        new_field = self.__copy__()
        o = getattr(other, 'F', other)

        new_field.F = self.F + o
        
        return new_field
    
    def __iadd__(self, other):
        o = getattr(other, 'F', other)
        self.F += o
        return self
    
    def __sub__(self, other):
        new_field = self.__copy__()
        o = getattr(other, 'F', other)

        new_field.F = self.F - o

        return new_field
    
    def __isub__(self, other):
        o = getattr(other, 'F', other)
        self.F -= o
        return self

    def __mul__(self, other):
        new_field = self.__copy__()

        o = getattr(other, 'F', other)
        new_field.F = self.F * o

        return new_field

    def __imul__(self, other):
        o = getattr(other, 'F', other)
        self.F *= o

        return self

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        new_field = self.__copy__()

        o = getattr(other, 'F', other)
        new_field.F = self.F / o

        return new_field
    
    def __itruediv__(self, other):
        o = getattr(other, 'F', other)
        self.F /= o

        return self

    def __pow__(self, power):
        new_field = self.__copy__()
        new_field.F = self.F ** power

        return new_field
    
    def __ipow__(self, power):
        self.F **= power
        return self
    
    def conj(self):
        self.F = self.F.conj()
        return self

    # TODO: more arithmetic

    # JIT compiling this didn't seem useful in initial tests, at least as written...
    #@partial(jax.jit, static_argnums=(1,))
    def nn_field(self, axis, shift=1):
        nn_shift = self.lattice.shift(self.F, axis=axis, shift=shift)

        # Apply boundary conditions globally with some arcane NumPy manipulations
        BC_factor = self.bc[axis]  # 1 or -1
        coords_ax = jnp.meshgrid(*[jnp.arange(Li) for Li in self.lattice.dims], indexing='ij')[axis]

        _, winding = jnp.divmod(coords_ax+shift, self.lattice.dims[axis])
        BC_field = (BC_factor)**(winding)

        LF = self.copy()
        LF.F = nn_shift * BC_field.reshape(self.st_dims)

        return LF
    
    def unit_fill(self):
        self.F = jnp.ones_like(self.F)
        return self
    
    def zero_fill(self):
        self.F = jnp.zeros_like(self.F)
        return self

    # Legacy function; will probably be removed in later version.
    def nn(self, coords, dir, backwards=False):
        # Get nn from lattice
        nn_raw = self.lattice.nn(coords, dir, backwards=backwards)

        # Apply boundary conditions
        new_C = []
        BC_factor = 1.0
        for ax in range(self.lattice.d):
            winding, site = divmod(nn_raw[ax], self.lattice.dims[ax])
            BC_factor *= float(self.bc[ax])**winding
            new_C.append(site)

        # Pass through extra coordinates (e.g. unit cell positions)
        if len(coords) > self.lattice.d:
            for i in range(self.lattice.d, len(coords)):
                new_C.append(nn_raw[i])

        return tuple(new_C), BC_factor
    
tree_util.register_pytree_node(
    OldLatticeField,
    OldLatticeField._tree_flatten,
    OldLatticeField._tree_unflatten,
)