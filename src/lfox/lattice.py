import cmath
import copy
import dataclasses
import enum
import math
import numbers
import operator
from abc import ABC, abstractmethod
from fractions import Fraction
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

    def __post_init__(self):
        # Dimensions of the physical lattice
        # May differ from space-time dims (e.g.
        # non-trivial unit cell)
        self._dims = self.st_dims

    @property
    def volume(self):
        # Number of lattice sites, including any unit-cell structure.
        return math.prod(self._dims)


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


# --- Boundary conditions ------------------------------------------------------
#
# A field carries one boundary condition per spacetime axis: a phase,
# psi(x + L) = exp(i theta) psi(x), or Dirichlet (no site past the edge).
# bc is static -- part of a field's pytree structure, which lax.cond and scan
# require to match exactly -- so phases are stored exactly, never as floats.

_EXACT = {Fraction(0): 1, Fraction(1, 2): 1j, Fraction(1): -1, Fraction(3, 2): -1j}


class Phase:
    """The boundary phase exp(i theta), theta = pi * n / d.

    theta is an exact fraction of pi, reduced mod 2 pi, so phases compare and
    hash exactly: Phase(1, 1) == -1, and Phase(1, 3)**3 is Phase(1, 1) rather
    than -1 + 4e-16j.  Build one with `phase(n, d)`.
    """

    __slots__ = ("theta",)

    def __init__(self, n, d=1):
        self.theta = Fraction(n, d) % 2  # in units of pi

    @property
    def value(self):
        # As a Python number; exact at multiples of pi/2, so a sign keeps a
        # real field real.
        return _EXACT.get(self.theta, cmath.exp(1j * math.pi * self.theta))

    def __mul__(self, other):
        if not isinstance(other, Phase):
            return NotImplemented
        return Phase(self.theta + other.theta)

    def __truediv__(self, other):
        if not isinstance(other, Phase):
            return NotImplemented
        return Phase(self.theta - other.theta)

    def __pow__(self, n):
        return Phase(self.theta * n)

    def conjugate(self):
        return Phase(-self.theta)

    def __complex__(self):
        return complex(self.value)

    def __eq__(self, other):
        if isinstance(other, Phase):
            return self.theta == other.theta
        if isinstance(other, numbers.Number):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(self.value)  # consistent with == against plain numbers

    def __repr__(self):
        # Signs print as plain numbers, so a bc reads (1, 1, -1).
        if self.theta in (0, 1):
            return repr(self.value)
        return f"phase({self.theta.numerator}, {self.theta.denominator})"


def phase(n, d=1):
    """The boundary phase exp(i pi n / d): phase(1, 1) is -1, phase(1, 2) is i."""
    return Phase(n, d)


class _Dirichlet(enum.Enum):
    DIRICHLET = "Dirichlet"

    def __repr__(self):
        return "DIRICHLET"


# No site past the edge: a hop off the edge gives 0.  Spelled 0 in bc.
DIRICHLET = _Dirichlet.DIRICHLET


def _boundary(b):
    # One user-supplied boundary condition, normalized.
    if b is DIRICHLET or isinstance(b, Phase):
        return b
    if isinstance(b, numbers.Number):
        if b == 0:
            return DIRICHLET
        if b == 1:
            return Phase(0)
        if b == -1:
            return Phase(1)
    raise ValueError(
        f"boundary condition {b!r} is not a phase or Dirichlet: use 1, -1, "
        "lat.phase(n, d) for exp(i pi n / d), or 0 for Dirichlet"
    )


def _boundaries(bc, d):
    if len(bc) == 0:
        return (Phase(0),) * d  # periodic
    if len(bc) != d:
        raise ValueError(
            f"need one boundary condition per spacetime axis ({d}), got {bc!r}"
        )
    return tuple(_boundary(b) for b in bc)


def _wrap_factor(b, winding):
    # The factor b**winding picked up by a value that wrapped `winding` times.
    if b is DIRICHLET:
        return 1 if winding == 0 else 0
    return (b**winding).value


# The product rule: how bc combines under each operator, axis by axis.  A raw
# scalar or array has no boundary condition (None) and leaves it unchanged;
# Dirichlet absorbs under *, / and **.


def _mul_bc(left, right, _):
    if left is None or right is None:
        return right if left is None else left
    return tuple(
        DIRICHLET if DIRICHLET in (a, b) else a * b for a, b in zip(left, right)
    )


def _div_bc(left, right, _):
    if right is None:
        return left
    inverse = tuple(b if b is DIRICHLET else Phase(0) / b for b in right)
    return inverse if left is None else _mul_bc(left, inverse, None)


def _sum_bc(left, right, other):
    bc = right if left is None else left
    if left is None or right is None:
        # A raw scalar is a constant, i.e. periodic; a raw array is taken on
        # trust (HMC adds raw momenta to fields).
        if jnp.ndim(other) == 0 and any(b not in (1, DIRICHLET) for b in bc):
            raise ValueError(f"cannot add a constant to a field with boundary {bc}")
        return bc
    if left != right:
        raise ValueError(
            f"cannot add fields with different boundary conditions {left} and {right}"
        )
    return bc


def _pow_bc(left, right, exponent):
    # An integer power acts on the phase.  Anything else -- a fractional,
    # traced or field-valued exponent -- has no well-defined phase unless the
    # base is periodic (or Dirichlet).
    integer = isinstance(exponent, numbers.Integral) or (
        isinstance(exponent, float) and exponent.is_integer()
    )
    if right is None and integer:
        return tuple(a if a is DIRICHLET else a ** int(exponent) for a in left)
    if right is not None and any(b not in (1, DIRICHLET) for b in right):
        raise ValueError(f"exponent field must be periodic, not boundary {right}")
    if any(a not in (1, DIRICHLET) for a in left):
        raise ValueError(
            f"non-integer power of a field with boundary {left} is ambiguous"
        )
    return _mul_bc(left, right, None)


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
        # Default periodic; otherwise validated and normalized (see Phase).
        self.bc = _boundaries(self.bc, len(self.lattice.st_dims))

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
    def _field_op(f, bc_rule, rev=False):
        # `other` may be a field, or a raw scalar or array (no bc: None).  The
        # bc rule sees both sides' bc in operator order, plus `other`'s values.
        def op(self, other):
            o = getattr(other, "F", other)
            other_bc = getattr(other, "bc", None)
            if rev:
                new_F, bc = f(o, self.F), bc_rule(other_bc, self.bc, o)
            else:
                new_F, bc = f(self.F, o), bc_rule(self.bc, other_bc, o)

            return self.copy_new_F(new_F, bc=bc)

        return op

    __add__ = _field_op(operator.add, _sum_bc)
    __radd__ = _field_op(operator.add, _sum_bc, rev=True)
    __sub__ = _field_op(operator.sub, _sum_bc)
    __rsub__ = _field_op(operator.sub, _sum_bc, rev=True)
    __mul__ = _field_op(operator.mul, _mul_bc)
    __rmul__ = _field_op(operator.mul, _mul_bc, rev=True)
    __truediv__ = _field_op(operator.truediv, _div_bc)
    __rtruediv__ = _field_op(operator.truediv, _div_bc, rev=True)
    __pow__ = _field_op(operator.pow, _pow_bc)

    def __neg__(self):
        return self.copy_new_F(-self.F)

    def conj(self):
        bc = tuple(b if b is DIRICHLET else b.conjugate() for b in self.bc)
        return self.copy_new_F(self.F.conj(), bc=bc)

    # TODO: more arithmetic?

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    def _nn_field(field, lattice, axis, shift, bc):
        # Site x receives the value from x - shift.  Writing shift = q L + r
        # with 0 <= r < L, sites x < r wrapped q + 1 times and the rest q
        # times, so the boundary factor is bc**-(q+1) on the first r slices and
        # bc**-q on the others.  (Signs show up at the target, not the source.)
        nn_shift = lattice.shift(field, axis=axis, shift=shift)
        q, r = divmod(shift, lattice.st_dims[axis])
        f_lo, f_hi = _wrap_factor(bc[axis], -q - 1), _wrap_factor(bc[axis], -q)

        def scaled(x, f):
            return x if f == 1 else jnp.zeros_like(x) if f == 0 else f * x

        if r == 0:
            return scaled(nn_shift, f_hi)
        lo, hi = jnp.split(nn_shift, [r], axis=axis)
        return jnp.concatenate([scaled(lo, f_lo), scaled(hi, f_hi)], axis=axis)

    def nn_field(self, axis, shift=1):
        shifted_field = self._nn_field(
            self.F,
            lattice=self.lattice,
            axis=axis,
            shift=shift,
            bc=self.bc,
        )

        return self.copy_new_F(shifted_field)

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)

        return new

    def copy_new_F(self, F, bc=None):
        # The same field with new values (and, optionally, boundary conditions).
        if bc is None:
            return dataclasses.replace(self, F=F)
        return dataclasses.replace(self, F=F, bc=bc)

    def copy(self):
        return self.__copy__()
