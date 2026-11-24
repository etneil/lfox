"""Design tests for gamma matrices and the Wilson operator, written before the
implementation.  The design is in `fermions_design.md`.

  * **A spinor is a plain `LatticeField`** with a `"spin"` axis, usually complex
    and antiperiodic in time.  There is no fermion subclass.
  * **`fermions.gammas(d)`** gives Euclidean, Hermitian gamma matrices with
    {g_mu, g_nu} = 2 delta_mu_nu, for d = 2, 3, 4: Pauli matrices (spin size 2)
    for d = 2, 3, and DeGrand-Rossi (spin size 4) for d = 4.  `gamma_mu` goes
    with lattice axis mu.  **`fermions.gamma5(d)`**, for even d, is Hermitian,
    squares to one and anticommutes with every gamma: sigma_3 = -i g0 g1 in
    d = 2, and g0 g1 g2 g3 = diag(1, 1, -1, -1) in d = 4.
  * **`fermions.wilson(psi, kappa)`** is the Wilson operator in hopping form,

        D psi = psi - kappa sum_mu [ (1 - g_mu) psi(x+mu) + (1 + g_mu) psi(x-mu) ],

    acting on the spin axis; any other axes (color, flavor) pass through.  The
    hop picks up the field's boundary conditions.  On a plane wave
    exp(ip.x) u it is the matrix 1 - 2 kappa sum_mu (cos p_mu - i g_mu sin p_mu)
    acting on u, with p_mu shifted by pi/L_mu on an antiperiodic axis.  It is
    gamma5-hermitian: g5 D g5 = D^dagger.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.fermions as fermions
import lfox.lattice as lat

DEGRAND_ROSSI = [
    [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
]
PAULI = [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]


def _spinor(key, dims, bc=(), spin=None, extra=None):
    spin = spin or (4 if len(dims) == 4 else 2)
    axes = {"spin": spin, **(extra or {})}
    shape = tuple(dims) + tuple(axes.values())
    k1, k2 = jax.random.split(jax.random.PRNGKey(key))
    F = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    return lat.LatticeField(lattice=lat.SquareLattice(st_dims=dims), F=F, axes=axes, bc=bc)


# --- gamma matrices -----------------------------------------------------------


@pytest.mark.parametrize("d, size", [(2, 2), (3, 2), (4, 4)])
def test_gammas_are_a_hermitian_clifford_basis(d, size):
    g = [np.asarray(m) for m in fermions.gammas(d)]
    assert len(g) == d
    I = np.eye(size)
    for mu, nu in itertools.product(range(d), repeat=2):
        assert g[mu].shape == (size, size)
        assert np.allclose(g[mu] @ g[nu] + g[nu] @ g[mu], 2 * (mu == nu) * I)
    for m in g:
        assert np.allclose(m, m.conj().T)


def test_gammas_are_pauli_in_two_and_three_dimensions():
    assert np.allclose(np.asarray(fermions.gammas(2)), PAULI[:2])
    assert np.allclose(np.asarray(fermions.gammas(3)), PAULI)


def test_gammas_are_degrand_rossi_in_four_dimensions():
    assert np.allclose(np.asarray(fermions.gammas(4)), DEGRAND_ROSSI)


@pytest.mark.parametrize("d", [2, 4])
def test_gamma5(d):
    g = [np.asarray(m) for m in fermions.gammas(d)]
    g5 = np.asarray(fermions.gamma5(d))
    I = np.eye(len(g5))
    assert np.allclose(g5, g5.conj().T)
    assert np.allclose(g5 @ g5, I)
    for m in g:
        assert np.allclose(g5 @ m + m @ g5, 0)


def test_gamma5_conventions():
    g = [np.asarray(m) for m in fermions.gammas(2)]
    assert np.allclose(fermions.gamma5(2), np.diag([1, -1]))
    assert np.allclose(fermions.gamma5(2), -1j * g[0] @ g[1])

    g = [np.asarray(m) for m in fermions.gammas(4)]
    assert np.allclose(fermions.gamma5(4), np.diag([1, 1, -1, -1]))
    assert np.allclose(fermions.gamma5(4), g[0] @ g[1] @ g[2] @ g[3])


@pytest.mark.parametrize("d", [1, 5])
def test_unsupported_dimensions_are_an_error(d):
    with pytest.raises(ValueError):
        fermions.gammas(d)


def test_there_is_no_gamma5_in_odd_dimensions():
    with pytest.raises(ValueError):
        fermions.gamma5(3)


# --- the Wilson operator ------------------------------------------------------


def _plane_wave(dims, n, bc, u):
    # psi(x) = exp(i p.x) u, with p allowed by the boundary conditions.
    bc = bc or (1,) * len(dims)
    p = [(2 * np.pi * n_mu + (np.pi if b == -1 else 0)) / L for n_mu, L, b in zip(n, dims, bc)]
    x = np.meshgrid(*[np.arange(L) for L in dims], indexing="ij")
    phase = np.exp(1j * sum(p_mu * x_mu for p_mu, x_mu in zip(p, x)))
    F = phase[..., None] * np.asarray(u)
    psi = lat.LatticeField(
        lattice=lat.SquareLattice(st_dims=dims), F=F, axes={"spin": len(u)}, bc=bc
    )
    return psi, p, phase


def _momentum_matrix(p, kappa):
    g = [np.asarray(m) for m in fermions.gammas(len(p))]
    I = np.eye(len(g[0]))
    return I - 2 * kappa * sum(np.cos(p_mu) * I - 1j * g_mu * np.sin(p_mu) for p_mu, g_mu in zip(p, g))


PLANE_WAVES = [
    ((4, 6), (1, 2), ()),
    ((4, 6), (0, 1), (-1, 1)),
    ((3, 2, 2, 4), (1, 0, 1, 2), (1, 1, 1, -1)),
]


@pytest.mark.parametrize("dims, n, bc", PLANE_WAVES, ids=["2d", "2d-anti", "4d-anti-t"])
def test_free_wilson_on_a_plane_wave(dims, n, bc):
    kappa = 0.13
    u = np.random.default_rng(0).standard_normal(4 if len(dims) == 4 else 2) + 0.5j
    psi, p, phase = _plane_wave(dims, n, bc, u)

    Dpsi = fermions.wilson(psi, kappa)
    assert Dpsi.axes == psi.axes
    assert Dpsi.bc == psi.bc
    expected = phase[..., None] * (_momentum_matrix(p, kappa) @ u)
    assert np.allclose(Dpsi.F, expected)


def test_free_wilson_spectrum_is_the_plane_wave_spectrum():
    # Every eigenvalue of D on a small antiperiodic lattice is an eigenvalue of
    # the momentum-space matrix at one of the allowed momenta, and vice versa.
    from lfox.linalg import dense

    dims, bc, kappa = (4, 3), (-1, 1), 0.2
    like = _spinor(0, dims, bc)
    D = dense(lambda psi: fermions.wilson(psi, kappa), like)

    expected = []
    for n in itertools.product(*[range(L) for L in dims]):
        _, p, _ = _plane_wave(dims, n, bc, np.ones(2))
        expected.extend(np.linalg.eigvals(_momentum_matrix(p, kappa)))

    got = np.sort_complex(np.round(np.linalg.eigvals(np.asarray(D)), 9))
    assert np.allclose(got, np.sort_complex(np.round(np.array(expected), 9)))


@pytest.mark.parametrize(
    "dims, bc", [((4, 3), (-1, 1)), ((2, 2, 2, 3), (1, 1, 1, -1))], ids=["2d", "4d"]
)
def test_wilson_is_gamma5_hermitian(dims, bc):
    from lfox.linalg import dense

    kappa = 0.17
    like = _spinor(1, dims, bc)
    D = np.asarray(dense(lambda psi: fermions.wilson(psi, kappa), like))
    n_sites = np.prod(dims)
    G5 = np.kron(np.eye(n_sites), np.asarray(fermions.gamma5(len(dims))))  # .F order: sites, then spin

    assert np.allclose(G5 @ D @ G5, D.conj().T)
    assert not np.allclose(D, D.conj().T)  # ...and D itself is not Hermitian


def test_wilson_at_zero_kappa_is_the_identity():
    psi = _spinor(2, (4, 6), (-1, 1))
    assert jnp.allclose(fermions.wilson(psi, 0.0).F, psi.F)


def test_wilson_acts_on_spin_and_passes_other_axes_through():
    kappa = 0.15
    psi = _spinor(3, (4, 6), (-1, 1), extra={"color": 3})
    Dpsi = fermions.wilson(psi, kappa)
    assert Dpsi.axes == psi.axes
    for c in range(3):
        one = lat.LatticeField(
            lattice=psi.lattice, F=psi.F[..., c], axes={"spin": 2}, bc=psi.bc
        )
        assert jnp.allclose(Dpsi.F[..., c], fermions.wilson(one, kappa).F)

    # The order the axes were declared in does not matter.
    swapped = lat.LatticeField(
        lattice=psi.lattice, F=jnp.swapaxes(psi.F, -1, -2),
        axes={"color": 3, "spin": 2}, bc=psi.bc,
    )
    assert jnp.allclose(jnp.swapaxes(fermions.wilson(swapped, kappa).F, -1, -2), Dpsi.F)


def test_wilson_needs_a_spin_axis_of_the_right_size():
    with pytest.raises(ValueError, match="spin"):
        fermions.wilson(_spinor(4, (4, 6), spin=4), 0.1)  # 2d needs spin 2
    plain = lat.LatticeField(lattice=lat.SquareLattice(st_dims=(4, 6)), F=1.0)
    with pytest.raises(ValueError, match="spin"):
        fermions.wilson(plain, 0.1)


def test_wilson_is_jittable_in_kappa():
    psi = _spinor(5, (4, 6), (-1, 1))
    f = jax.jit(fermions.wilson)
    for kappa in (0.1, 0.2):
        assert jnp.allclose(f(psi, kappa).F, fermions.wilson(psi, kappa).F)
