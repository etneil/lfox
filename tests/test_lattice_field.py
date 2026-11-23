"""Design tests for `LatticeField` structure, written before the implementation.

These pin what a field *is*: its shape, its internal axes, its boundary
conditions, and how all three behave under arithmetic and hops.

  * **Shape.**  `F` is a scalar (shorthand for a constant field) or exactly
    the field's shape: the spacetime dims, then the internal axes in declared
    order.  Anything else is a `ValueError`, never a silent broadcast.  An
    integer `F` gives a float field.
  * **Named internal axes.**  `axes={"spin": 4}` (or the equivalent tuple of
    pairs) declares internal axes; `field.axes` is the tuple of
    `(name, size)` pairs.  `.F` is the user-facing view, in the declared
    order.  How the data is *stored* is private and may change (e.g. internal
    axes leading, for performance), so nothing outside `LatticeField` may
    depend on it; these tests only ever look at `.F`.
  * **Name-aligned arithmetic.**  Two fields must share a lattice.  Their
    axes are matched by name, never by position: an axis on one side only is
    broadcast, so `phi * psi` multiplies every spin component by `phi`.  The
    result has the left operand's axes, then any extra axes of the right
    operand in its order.  One name with two sizes is an error.  A raw array
    must be a scalar or exactly `.F.shape`.
  * **Boundary conditions** are one entry per spacetime axis, each a *phase*
    or *Dirichlet*:
      - spellings: `1`, `-1`, `lat.phase(n, d)` = exp(i theta) with
        theta = pi n / d -- the same theta as `twist` takes, so
        `lat.phase(1, 2)` is i -- and `0` for Dirichlet.  Anything else is a
        `ValueError`.
      - theta is stored as an exact fraction of pi, mod 2 pi, so
        `lat.phase(1, 1) == -1` exactly, and fields with equal boundary conditions have identical
        pytree structure.  `jax.lax.cond` and `scan` (HMC accept/reject,
        `evolve_many`) require that; floating-point phases would break it.
      - a hop sees psi(x + L) = bc * psi(x).  Dirichlet is not the number
        zero but "no site past the edge": a hop off the edge gives 0, while
        no real site is zero, so dividing by a Dirichlet field is fine.
      - under arithmetic, `*`, `/`, `**` and `conj` act on the phases (the
        product rule); Dirichlet absorbs under all of them.  A non-integer
        power has no single boundary phase (the branch cut makes it vary
        site by site), so it needs a periodic or Dirichlet field.  `+` and
        `-` require equal boundary conditions.  A raw scalar is a constant,
        i.e. periodic: neutral under `*` and `/`, and addable only to
        periodic or Dirichlet axes.  A raw array is taken on trust.  The
        invariant behind all of it: hopping a product is the product of the
        hops.
  * **Twist.**  `nn_field(axis, shift, twist=theta)` applies an extra
    boundary phase exp(i theta) on top of `bc`.  `theta` is traced, so it can
    be differentiated and changing it does not recompile.  Twisting a
    Dirichlet axis is an error.
"""

import cmath
import math
import operator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat

L1 = lat.SquareLattice(st_dims=(4,))
F1 = jnp.array([1.0, 2.0, 3.0, 4.0])


def _bc(spec):
    # Boundary spellings for parametrize, built lazily so that a missing
    # `lat.phase` fails the test rather than the whole module's collection.
    return lat.phase(*spec) if isinstance(spec, tuple) else spec


def _rand(key, shape, complex_=False):
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    x = 1.5 + jax.random.uniform(k1, shape)  # bounded away from 0, for division
    if complex_:
        x = x + 1j * jax.random.uniform(k2, shape)
    return x


# --- the shape of F -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(4,), (4, 4), (4, 1, 4), (1, 4, 4, 4), (2, 4, 4, 4)])
def test_rejects_a_wrongly_shaped_array(lat4, shape):
    with pytest.raises(ValueError, match="has shape"):
        lat.LatticeField(lattice=lat4, F=jnp.ones(shape))


def test_arithmetic_cannot_change_its_shape(lat4):
    # Broadcasting in an operator must not quietly turn a field into a batch
    # of fields.
    phi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)))
    with pytest.raises(ValueError, match="has shape"):
        phi * jnp.ones((2, 4, 4, 4))


def test_from_a_scalar_is_a_float_field(lat4):
    # F=1 is the shorthand for a unit field.  An integer F -- scalar or array
    # -- still gives a float field, or autodiff (dS) would refuse it.
    phi = lat.LatticeField(lattice=lat4, F=1)
    assert phi.F.shape == (4, 4, 4)
    assert jnp.issubdtype(phi.F.dtype, jnp.floating)
    assert jnp.all(phi.F == 1)

    ints = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4), dtype=int))
    assert jnp.issubdtype(ints.F.dtype, jnp.floating)


# --- named internal axes ------------------------------------------------------


def test_axes_are_part_of_the_shape(lat4):
    psi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4, 2)), axes={"spin": 2})
    assert psi.F.shape == (4, 4, 4, 2)
    assert psi.dims() == (4, 4, 4, 2)
    assert psi.d() == 3  # spacetime only
    assert psi.axes == (("spin", 2),)

    # A spacetime-only F is rejected rather than broadcast across the axis.
    with pytest.raises(ValueError, match="has shape"):
        lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4)), axes={"spin": 2})


def test_axes_spellings_are_equivalent(lat4):
    F = jnp.ones((4, 4, 4, 4, 3))
    a = lat.LatticeField(lattice=lat4, F=F, axes={"spin": 4, "color": 3})
    b = lat.LatticeField(lattice=lat4, F=F, axes=(("spin", 4), ("color", 3)))
    assert a.axes == b.axes == (("spin", 4), ("color", 3))
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_scalar_times_spinor_aligns_by_name(lat4):
    # The bug this replaces: with trailing positional indices, phi(x, y, z)
    # lined up against psi's (y, z, spin) and gave wrong numbers silently.
    phi = lat.LatticeField(lattice=lat4, F=_rand(0, (4, 4, 4)))
    psi = lat.LatticeField(lattice=lat4, F=_rand(1, (4, 4, 4, 4)), axes={"spin": 4})
    expected = psi.F * phi.F[..., None]

    for prod in (psi * phi, phi * psi):
        assert prod.axes == (("spin", 4),)
        assert jnp.allclose(prod.F, expected)

    assert jnp.allclose((psi + phi).F, psi.F + phi.F[..., None])
    assert jnp.allclose((phi - psi).F, phi.F[..., None] - psi.F)


def test_name_alignment_is_static_under_jit(lat4):
    phi = lat.LatticeField(lattice=lat4, F=_rand(0, (4, 4, 4)))
    psi = lat.LatticeField(lattice=lat4, F=_rand(1, (4, 4, 4, 4)), axes={"spin": 4})
    jitted = jax.jit(lambda a, b: a * b)(phi, psi)
    assert jitted.axes == (("spin", 4),)
    assert jnp.allclose(jitted.F, (phi * psi).F)


def test_same_axes_in_a_different_order_align_by_name(lat4):
    F = _rand(2, (4, 4, 4, 4, 3))
    a = lat.LatticeField(lattice=lat4, F=F, axes={"spin": 4, "color": 3})
    b = lat.LatticeField(
        lattice=lat4, F=jnp.swapaxes(F, -1, -2), axes={"color": 3, "spin": 4}
    )

    diff = a - b  # the same field, declared in the other order
    assert diff.axes == a.axes
    assert jnp.allclose(diff.F, 0.0)
    assert (b - a).axes == b.axes


def test_disjoint_axes_give_an_outer_product(lat4):
    psi = lat.LatticeField(lattice=lat4, F=_rand(3, (4, 4, 4, 4)), axes={"spin": 4})
    chi = lat.LatticeField(lattice=lat4, F=_rand(4, (4, 4, 4, 3)), axes={"color": 3})

    sc = psi * chi
    assert sc.axes == (("spin", 4), ("color", 3))
    assert jnp.allclose(sc.F, psi.F[..., :, None] * chi.F[..., None, :])

    cs = chi * psi
    assert cs.axes == (("color", 3), ("spin", 4))
    assert jnp.allclose(cs.F, jnp.swapaxes(sc.F, -1, -2))


def test_one_axis_name_with_two_sizes_is_an_error(lat4):
    a = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4, 4)), axes={"spin": 4})
    b = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4, 2)), axes={"spin": 2})
    with pytest.raises(ValueError, match="axis"):
        a * b


def test_fields_on_different_lattices_do_not_mix(lat4):
    phi = lat.LatticeField(lattice=lat4, F=1.0)
    other = lat.LatticeField(lattice=lat.SquareLattice(st_dims=(4, 4, 5)), F=1.0)
    with pytest.raises(ValueError, match="lattice"):
        phi + other


def test_a_raw_array_must_match_the_whole_field(lat4):
    # A raw array has no axis names, so it gets no broadcasting: a (4,) array
    # against a spinor would otherwise land on whichever axis is last.
    psi = lat.LatticeField(lattice=lat4, F=jnp.ones((4, 4, 4, 4)), axes={"spin": 4})
    with pytest.raises(ValueError, match="has shape"):
        psi * jnp.arange(4.0)
    with pytest.raises(ValueError, match="has shape"):
        psi * jnp.ones((4, 4, 4))

    full = jnp.full((4, 4, 4, 4), 2.0)
    assert jnp.allclose((psi * full).F, 2.0)


def test_hops_move_spacetime_only(lat4):
    F = _rand(5, (4, 4, 4, 4))
    psi = lat.LatticeField(lattice=lat4, F=F, axes={"spin": 4}, bc=(1, 1, -1))
    for ax in range(3):
        expected = jnp.roll(F, 1, axis=ax)
        if ax == 2:
            expected = expected.at[:, :, 0, :].multiply(-1)  # every spin component
        assert jnp.allclose(psi.nn_field(ax, 1).F, expected)


# --- boundary conditions: values ---------------------------------------------


def test_phases_are_exact():
    assert lat.phase(1, 1) == -1
    assert lat.phase(0, 1) == 1
    assert lat.phase(2, 4) == lat.phase(1, 2)  # reduced: both are i
    assert lat.phase(2, 1) == 1  # a full turn
    assert lat.phase(-1, 2) == lat.phase(3, 2)  # angles are mod 2 pi
    assert lat.phase(1, 3) != lat.phase(2, 3)
    assert complex(lat.phase(1, 2)) == 1j


@pytest.mark.parametrize("bad", [(2,), (0.5,), (cmath.exp(1j * math.pi / 3),)])
def test_a_boundary_condition_is_a_phase_or_dirichlet(bad):
    with pytest.raises(ValueError, match="boundary"):
        lat.LatticeField(lattice=L1, F=F1, bc=bad)


def test_one_boundary_condition_per_spacetime_axis(lat4):
    with pytest.raises(ValueError, match="boundary"):
        lat.LatticeField(lattice=lat4, F=1.0, bc=(1, -1))


def test_spellings_of_one_boundary_condition_are_one_structure():
    a = lat.LatticeField(lattice=L1, F=F1, bc=(-1,))
    b = lat.LatticeField(lattice=L1, F=F1, bc=(lat.phase(1, 1),))
    c = lat.LatticeField(lattice=L1, F=F1, bc=(-1.0,))
    assert a.bc == b.bc == c.bc == (-1,)
    assert (
        jax.tree_util.tree_structure(a)
        == jax.tree_util.tree_structure(b)
        == jax.tree_util.tree_structure(c)
    )

    d = lat.LatticeField(lattice=L1, F=F1, bc=(0,))
    e = lat.LatticeField(lattice=L1, F=F1, bc=(0.0,))
    assert d.bc == e.bc == (lat.DIRICHLET,)


def test_a_computed_phase_survives_lax_cond():
    # exp(i pi/3)**3 is -1 only up to rounding in floating point; stored as an
    # exact fraction of pi it is -1, so the two branches have one structure.
    a = lat.LatticeField(lattice=L1, F=F1, bc=(lat.phase(1, 3),))
    cubed = a * a * a
    plain = lat.LatticeField(lattice=L1, F=F1, bc=(-1,))
    assert cubed.bc == (-1,)
    assert jax.tree_util.tree_structure(cubed) == jax.tree_util.tree_structure(plain)
    out = jax.lax.cond(True, lambda: cubed, lambda: plain)
    assert jnp.allclose(out.F, F1**3)


# --- boundary conditions: hops ------------------------------------------------


def test_hop_with_a_complex_phase():
    # psi(x + L) = i psi(x), so psi(-1) = -i psi(3) and psi(4) = i psi(0).
    psi = lat.LatticeField(lattice=L1, F=F1, bc=(lat.phase(1, 2),))
    assert jnp.allclose(psi.nn_field(0, 1).F, jnp.array([-4j, 1, 2, 3]))
    assert jnp.allclose(psi.nn_field(0, -1).F, jnp.array([2, 3, 4, 1j]))
    assert jnp.allclose(psi.nn_field(0, 4).F, -1j * F1)


def test_a_real_field_stays_real_under_real_boundary_conditions():
    # A sign or a Dirichlet edge must not turn a real field complex, as a
    # cmath.exp(i pi) = -1 + 1.2e-16j factor would.
    for bc in [(-1,), (lat.phase(1, 1),), (0,)]:
        psi = lat.LatticeField(lattice=L1, F=F1, bc=bc)
        for shift in (1, -1, 4):
            assert psi.nn_field(0, shift).F.dtype == F1.dtype
    anti = lat.LatticeField(lattice=L1, F=F1, bc=(lat.phase(1, 2),))
    assert (anti * anti).nn_field(0, 1).F.dtype == F1.dtype  # i * i = -1, real


@pytest.mark.parametrize("zero", [0, 0.0])
def test_dirichlet_hop_off_the_edge_is_zero(zero):
    # With a float 0 the old winding factor computed 0**-1 = inf.
    psi = lat.LatticeField(lattice=L1, F=F1, bc=(zero,))
    assert jnp.allclose(psi.nn_field(0, 1).F, jnp.array([0.0, 1, 2, 3]))
    assert jnp.allclose(psi.nn_field(0, -1).F, jnp.array([2.0, 3, 4, 0]))
    assert jnp.allclose(psi.nn_field(0, 2).F, jnp.array([0.0, 0, 1, 2]))
    assert jnp.all(jnp.isfinite(psi.nn_field(0, 1).F))


# --- boundary conditions: arithmetic ------------------------------------------

PHASES = [1, -1, (1, 2), (1, 3)]


@pytest.mark.parametrize("op", [operator.mul, operator.truediv], ids=["mul", "div"])
@pytest.mark.parametrize("bc_a", PHASES, ids=str)
@pytest.mark.parametrize("bc_b", PHASES, ids=str)
@pytest.mark.parametrize("shift", [1, -1])
def test_hop_of_a_product_is_the_product_of_hops(op, bc_a, bc_b, shift):
    L = lat.SquareLattice(st_dims=(5,))
    a = lat.LatticeField(lattice=L, F=_rand(6, (5,), complex_=True), bc=(_bc(bc_a),))
    b = lat.LatticeField(lattice=L, F=_rand(7, (5,), complex_=True), bc=(_bc(bc_b),))
    lhs = op(a, b).nn_field(0, shift).F
    rhs = op(a.nn_field(0, shift).F, b.nn_field(0, shift).F)
    assert jnp.allclose(lhs, rhs)


@pytest.mark.parametrize("bc_a", PHASES, ids=str)
@pytest.mark.parametrize("unary", ["conj", "square", "inverse", "neg"])
def test_hop_commutes_with_unary_operations(bc_a, unary):
    f = {
        "conj": lambda x: x.conj(),
        "square": lambda x: x**2,
        "inverse": lambda x: x**-1,
        "neg": lambda x: -x,
    }[unary]
    L = lat.SquareLattice(st_dims=(5,))
    a = lat.LatticeField(lattice=L, F=_rand(8, (5,), complex_=True), bc=(_bc(bc_a),))
    for shift in (1, -1):
        assert jnp.allclose(f(a).nn_field(0, shift).F, f(a.nn_field(0, shift).F))


def test_product_rule_on_the_bc_itself(lat4):
    a = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, -1, lat.phase(1, 2)))
    assert (a * a).bc == (1, 1, -1)
    assert (a / a).bc == (1, 1, 1)
    assert a.conj().bc == (1, -1, lat.phase(-1, 2))
    assert (a**3).bc == (1, -1, lat.phase(3, 2))
    assert (a.conj() * a).bc == (1, 1, 1)


def test_a_non_integer_power_needs_a_periodic_field():
    # sqrt of an antiperiodic field has no well-defined boundary phase.  An
    # integral float is an integer power; Dirichlet absorbs as usual.
    L = lat.SquareLattice(st_dims=(5,))
    F = _rand(11, (5,))
    periodic = lat.LatticeField(lattice=L, F=F)
    anti = lat.LatticeField(lattice=L, F=F, bc=(-1,))
    dirichlet = lat.LatticeField(lattice=L, F=F, bc=(0,))

    assert (periodic**0.5).bc == (1,)
    assert (anti**2.0).bc == (1,)
    assert (dirichlet**0.5).bc == (lat.DIRICHLET,)
    assert (periodic**periodic).bc == (1,)
    with pytest.raises(ValueError, match="boundary"):
        anti**0.5
    with pytest.raises(ValueError, match="boundary"):
        periodic**anti


def test_signs_print_as_numbers(lat4):
    phi = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, -1, lat.phase(1, 3)))
    assert repr(phi.bc) == "(1, -1, phase(1, 3))"


def test_sums_require_equal_boundary_conditions(lat4):
    a = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, 1, -1))
    b = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, 1, 1))
    c = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, 1, lat.phase(1, 1)))
    with pytest.raises(ValueError, match="boundary"):
        a + b
    with pytest.raises(ValueError, match="boundary"):
        a - b
    assert (a + c).bc == (1, 1, -1)


def test_raw_scalars_and_arrays_are_neutral(lat4):
    # A raw array carries no boundary condition and is taken on trust (HMC
    # adds raw momenta to fields).  A scalar is a constant -- a periodic field
    # -- so it is neutral under * and /.
    a = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, 1, -1))
    ones = jnp.ones((4, 4, 4))
    for result in (2.0 * a, 6.0 / a, a * ones, a + ones, ones - a):
        assert result.bc == (1, 1, -1)


def test_a_constant_adds_only_to_periodic_or_dirichlet_axes(lat4):
    # A constant is periodic, so adding one to an antiperiodic field leaves no
    # consistent boundary condition.
    a = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, 1, -1))
    for add in (lambda: a + 1.0, lambda: 1.0 - a, lambda: a - jnp.array(2.0)):
        with pytest.raises(ValueError, match="boundary"):
            add()

    p = lat.LatticeField(lattice=lat4, F=1.0, bc=(1, 0, 1))
    assert (p + 1.0).bc == (1, lat.DIRICHLET, 1)
    assert (1.0 - p).bc == (1, lat.DIRICHLET, 1)


def test_dirichlet_absorbs_and_division_is_fine():
    # No real site is zero, so dividing by a Dirichlet field is well defined.
    L = lat.SquareLattice(st_dims=(5,))
    d = lat.LatticeField(lattice=L, F=_rand(9, (5,)), bc=(0,))
    p = lat.LatticeField(lattice=L, F=_rand(10, (5,)), bc=(-1,))

    for result in (d * p, p * d, d / p, p / d, d**2, d**-1, d.conj()):
        assert result.bc == (lat.DIRICHLET,)
        assert jnp.all(jnp.isfinite(result.F))
        assert result.nn_field(0, 1).F[0] == 0
        assert result.nn_field(0, -1).F[-1] == 0

    assert jnp.allclose((p / d).F, p.F / d.F)
    assert (d + d).bc == (lat.DIRICHLET,)


def test_dirichlet_and_a_phase_do_not_add():
    d = lat.LatticeField(lattice=L1, F=F1, bc=(0,))
    p = lat.LatticeField(lattice=L1, F=F1, bc=(1,))
    with pytest.raises(ValueError, match="boundary"):
        d + p


# --- twist --------------------------------------------------------------------


def test_twist_is_an_extra_boundary_phase():
    # The same theta: twist=pi/2 and phase(1, 2) are both i.
    periodic = lat.LatticeField(lattice=L1, F=F1)
    plus_i = lat.LatticeField(lattice=L1, F=F1, bc=(lat.phase(1, 2),))
    for shift in (1, -1, 4):
        assert jnp.allclose(
            periodic.nn_field(0, shift, twist=math.pi / 2).F, plus_i.nn_field(0, shift).F
        )

    # Stacks on top of bc: -1 * i = -i = phase(3, 2).
    anti = lat.LatticeField(lattice=L1, F=F1, bc=(-1,))
    minus_i = lat.LatticeField(lattice=L1, F=F1, bc=(lat.phase(3, 2),))
    assert jnp.allclose(anti.nn_field(0, 1, twist=math.pi / 2).F, minus_i.nn_field(0, 1).F)


def test_twist_is_traced_and_differentiable():
    psi = lat.LatticeField(lattice=L1, F=F1)
    traces = []

    def wrapped(theta):
        # After a +1 hop, site 0 holds exp(-i theta) * psi(3) = 4 exp(-i theta).
        traces.append(1)
        return jnp.real(psi.nn_field(0, 1, twist=theta).F[0])

    f = jax.jit(wrapped)
    assert np.isclose(f(0.3), 4 * math.cos(0.3))
    assert np.isclose(f(0.7), 4 * math.cos(0.7))
    assert len(traces) == 1  # a new theta does not recompile

    assert np.isclose(jax.grad(wrapped)(0.3), -4 * math.sin(0.3))


def test_twisting_a_dirichlet_axis_is_an_error():
    d = lat.LatticeField(lattice=L1, F=F1, bc=(0,))
    with pytest.raises(ValueError, match="Dirichlet"):
        d.nn_field(0, 1, twist=0.1)
