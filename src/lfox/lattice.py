import numpy as np

class Lattice:
    
    def __init__(self, dims):

        self.dims = dims
        self.d = len(dims)

    # TODO: implement a function that will accept out-of-bounds indices,
    # and appropriately apply boundary conditions.


class LatticeField:

    def __init__(self, lattice, dtype=np.float):
        self.lattice = lattice
        self.dtype = dtype

        # Initialize the field
        self.field = np.zeros(lattice.dims, dtype=self.dtype)

        