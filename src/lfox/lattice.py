import numpy as np
from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from jax import tree_util
from functools import partial
import copy

class Lattice(ABC):
    
    def __init__(self, dims):

        self.dims = dims
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
    Lattice,
    Lattice._tree_flatten,
    Lattice._tree_unflatten,
)

class SquareLattice(Lattice):

    def shift(self, field, axis, shift=1):
        return jnp.roll(field, shift=shift, axis=axis)
    
class HoneycombLattice(Lattice):

    def __init__(self, dims):
        # A/B unit cell structure, automatically in the dim-0 direction.
        # Note that this means that the number of lattice sites in dimension
        # 0 is actually 2*dims[0].

        self.unit_cell = [0, 1]
        super().__init__(dims=dims)


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


class LatticeField:

    def __init__(self, lattice: Lattice, field=None, bc=None, dtype=float):
        self.lattice = lattice
        self.dtype = dtype

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
        if field is None:
            if hasattr(lattice, 'unit_cell'):
                dims = lattice.dims + [ len(lattice.unit_cell) ]
            else:
                dims = lattice.dims
            self.field = jnp.zeros(dims, dtype=self.dtype)
        else:
            self.field = field

    def _tree_flatten(self):
        children = (self.field,)
        aux_data = {
            'lattice': self.lattice,
            'bc': self.bc,
            'dtype': self.dtype,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(lattice=aux_data['lattice'], field=children[0], bc=aux_data['bc'], dtype=aux_data['dtype'])

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        # JAX/NumPy arrays are immutable, and this was slow for some reason
        # new.field = jnp.copy(self.field)
        new.field = self.field  

        return new
    
    def copy(self):
        return self.__copy__()


    # Arithmetic with fields - pass through to the field array
    def __add__(self, other):
        new_field = self.__copy__()
        o = getattr(other, 'field', other)

        new_field.field = self.field + o
        
        return new_field
    
    def __iadd__(self, other):
        o = getattr(other, 'field', other)
        self.field += o
        return self
    
    def __sub__(self, other):
        new_field = self.__copy__()
        o = getattr(other, 'field', other)

        new_field.field = self.field - o

        return new_field
    
    def __isub__(self, other):
        o = getattr(other, 'field', other)
        self.field -= o
        return self

    def __mul__(self, other):
        new_field = self.__copy__()

        o = getattr(other, 'field', other)
        new_field.field = self.field * o

        return new_field

    def __imul__(self, other):
        o = getattr(other, 'field', other)
        self.field *= o

        return self

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        new_field = self.__copy__()
        new_field.field = self.field ** power

        return new_field
    
    def __ipow__(self, power):
        self.field **= power
        return self

    # TODO: more arithmetic

    # JIT compiling this didn't seem useful in initial tests, at least as written...
    #@partial(jax.jit, static_argnums=(1,))
    def nn_field(self, axis, shift=1):
        nn_shift = self.lattice.shift(self.field, axis=axis, shift=shift)

        # Apply boundary conditions globally with some arcane NumPy manipulations
        BC_factor = self.bc[axis]  # 1 or -1
        coords_ax = jnp.meshgrid(*[jnp.arange(Li) for Li in self.lattice.dims], indexing='ij')[axis]

        _, winding = jnp.divmod(coords_ax+shift, self.lattice.dims[axis])
        BC_field = (BC_factor)**(winding)

        LF = self.copy()
        LF.field = nn_shift * BC_field

        return LF

#        return nn_shift * BC_field

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
    LatticeField,
    LatticeField._tree_flatten,
    LatticeField._tree_unflatten,
)