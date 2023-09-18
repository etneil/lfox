import numpy as np
from abc import ABC, abstractmethod
import jax.numpy as jnp

class Lattice(ABC):
    
    def __init__(self, dims):

        self.dims = dims
        self.d = len(dims)

    # TODO: implement a function that will accept out-of-bounds indices,
    # and appropriately apply boundary conditions.

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


# So far, not working with JIT.
# Look at rewriting using PyTrees, e.g. https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
# and https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees ???
class LatticeField:

    def __init__(self, lattice, bc=None, dtype=float):
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
        self.field = jnp.zeros(lattice.dims, dtype=self.dtype)

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

        return new_C, BC_factor
        return self.field[tuple(new_C)] * BC_factor