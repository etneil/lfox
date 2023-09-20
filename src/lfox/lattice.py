import numpy as np
from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax

class Lattice(ABC):
    
    def __init__(self, dims):

        self.dims = dims
        self.d = len(dims)

    # TODO: change this to a "shift" function, that will take an arbitrary
    # array with dimensions matching self.dims and shift it appropriately.
    # Boundary conditions to be applied within LatticeField objects
    # (since two fields on the same lattice can have different BCs.)

    @abstractmethod
    def nn(self, dir):
        pass

class SquareLattice(Lattice):

    def nn(self, coords, dir, backwards=False):
        assert len(coords) == self.d
        disp = -1 if backwards else 1

        new_C = []
        for ax, C in enumerate(coords):
            new_C.append(C+disp if ax == dir else C)
        
        return (new_C)

class HoneycombLattice(Lattice):

    def __init__(self, dims):
        # A/B unit cell structure, automatically in the dim-0 direction.
        # Note that this means that the number of lattice sites in dimension
        # 0 is actually 2*dims[0].

        self.unit_cell = [0, 1]
        super().__init__(dims=dims)

    def nn(self, coords, dir, backwards=False):
        assert len(coords) == self.d + 1
        disp = -1 if backwards else 1
        new_unit = (coords[-1] + 1) % 2

        new_C = []
        for ax, C in enumerate(coords):
            new_C.append(C+disp if ax == dir else C)
        new_C.append(new_unit)

        return (new_C)


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


    def nn_field(self, axis, shift=1):
        if axis == 0 and hasattr(self.lattice, 'unit_cell'):
            # Deal with unit cell by rolling in extra dimension, too
            nn_raw = jnp.roll(self.field, shift=shift, axis=0)
            nn_raw = jnp.roll(nn_raw, shift=shift, axis=-1)
        else:
            nn_raw = jnp.roll(self.field, shift=shift, axis=axis)

        # Apply boundary conditions
        BC_factor = self.bc[axis]  # 1 or -1
        coords_ax = jnp.meshgrid(*[jnp.arange(Li) for Li in self.lattice.dims], indexing='ij')[axis]

        _, winding = jnp.divmod(coords_ax+shift, self.lattice.dims[axis])
        BC_field = (BC_factor)**(winding)

        return nn_raw * BC_field







        

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
        return self.field[tuple(new_C)] * BC_factor
    
from jax import tree_util
tree_util.register_pytree_node(
    LatticeField,
    LatticeField._tree_flatten,
    LatticeField._tree_unflatten,
)