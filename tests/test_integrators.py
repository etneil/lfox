"""Design tests for nested (multi-timescale) MD integration, written before the
implementation.

An `Integrator(tau, levels)` integrates Hamilton's equations for a trajectory
of length `tau`.  Its levels are listed outermost first; each is a
`Level(scheme, Nstep, terms)`, with the names of the action terms whose force
it applies.  Cheap, stiff forces belong on inner levels, expensive, smooth ones
(a pseudofermion) on outer ones.

  * **A scheme is a tuple of steps** `(op, c)`.  Op "P" kicks the momenta with
    the level's force for time c h; op "Q" advances the positions for time c h,
    where h is the level's step size.  A scheme must be symmetric (a
    palindrome, so that it is reversible), and its P and its Q coefficients
    must each sum to 1.  Provided: velocity-first `Leapfrog` (PQP) and
    `Omelyan` (PQPQP), and position-first `Leapfrog_QPQ` and `Omelyan_QPQ`.
    Both Omelyan orderings use lambda = 0.1931833 (lambda_c of Takaishi and
    de Forcrand, hep-lat/0505020); the optimum varies slightly with the theory,
    its parameters and the ordering, so to tune lambda, write the tuple.
  * **Nesting**: a Q step of level k runs level k + 1 over its interval; only
    the innermost level's Q steps move the fields.  The outermost step is
    h_0 = tau / N_0, and a Q step of coefficient c at level k runs N_{k+1}
    steps of h_{k+1} = c h_k / N_{k+1}.
  * **Force reuse**: a scheme that begins and ends with P evaluates its force
    at the start of the trajectory, and reuses the force from the end of each
    step at the start of the next -- also across consecutive runs of an inner
    level, between which only outer kicks happen and the fields do not move.
    So Leapfrog costs N + 1 force evaluations, not 2N, and Omelyan 2N + 1,
    not 3N.
  * `tau` is the integrator's only pytree leaf (so it is traced, and may vary
    across a MultiChain); the schemes and every `Nstep` are static.
  * `integrate(delta_X, delta_P, X, P)` takes a tuple `delta_P` with one force
    closure per level, outermost first, each returning (dP, info).  It returns
    (X, P, info), with each level's info stacked over that level's force
    evaluations in time order, and the levels' info dicts merged.
  * **HMC** builds each level's force from the terms it names.  Every term of
    the action must be named by exactly one level; a lone level may leave
    `terms` out, and then applies the whole action.  Violations raise at HMC
    construction.
  * `LeapfrogIntegrator(tau, Nstep)` and `OmelyanIntegrator(tau, Nstep)` are
    one-level shorthands, velocity-first.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lfox.lattice as lat
from lfox.action import Action, Term
from lfox.evolution import (
    HMC,
    Integrator,
    Leapfrog,
    Leapfrog_QPQ,
    LeapfrogIntegrator,
    Level,
    Omelyan,
    Omelyan_QPQ,
    OmelyanIntegrator,
)
from tests.phi4 import ScalarTerm

LAM = 0.1931833
LAT = lat.SquareLattice(st_dims=(4, 4))
SCHEMES = {
    "Leapfrog": Leapfrog,
    "Leapfrog_QPQ": Leapfrog_QPQ,
    "Omelyan": Omelyan,
    "Omelyan_QPQ": Omelyan_QPQ,
}


def _field(key, low=-1.0, high=1.0):
    F = jax.random.uniform(jax.random.PRNGKey(key), (4, 4), minval=low, maxval=high)
    return lat.LatticeField(lattice=LAT, F=F)


X0 = {"x": _field(0)}
P0 = {"x": _field(1)}
W2 = _field(2, 0.5, 2.0)  # a different frequency at every site


# Hamilton's equations for H = p^2/2 + w2 x^2/2 + x^4/4, split into forces.


def drift(X, P):
    return P


def harmonic(X, P):
    return {"x": -(W2 * X["x"])}, {}


def half_harmonic(X, P):
    return {"x": -0.5 * (W2 * X["x"])}, {}


def quartic(X, P):
    return {"x": -(X["x"] ** 3)}, {}


def full(X, P):
    return {"x": -(W2 * X["x"]) - X["x"] ** 3}, {}


def zero(X, P):
    return {"x": 0.0 * X["x"]}, {}


def _integrator(outer, N, inner=(), tau=1.0):
    # The outer scheme with N steps, then fixed inner levels [(scheme, Nstep)].
    return Integrator(
        tau=tau, levels=[Level(outer, N)] + [Level(s, n) for s, n in inner]
    )


# name: (outer scheme, inner levels, forces outermost first).  Each test
# chooses the outer Nstep; scaling it scales every level's step size.
CASES = {
    "Leapfrog": (Leapfrog, (), (full,)),
    "Leapfrog_QPQ": (Leapfrog_QPQ, (), (full,)),
    "Omelyan": (Omelyan, (), (full,)),
    "Omelyan_QPQ": (Omelyan_QPQ, (), (full,)),
    "Omelyan/Leapfrog": (Omelyan, [(Leapfrog, 2)], (quartic, harmonic)),
    "Leapfrog_QPQ/Omelyan_QPQ": (Leapfrog_QPQ, [(Omelyan_QPQ, 3)], (quartic, harmonic)),
    "three levels": (
        Leapfrog,
        [(Omelyan, 2), (Leapfrog_QPQ, 2)],
        (quartic, half_harmonic, half_harmonic),
    ),
}


# --- Schemes and levels -------------------------------------------------------


def test_schemes_are_the_standard_sequences():
    assert Leapfrog == (("P", 0.5), ("Q", 1.0), ("P", 0.5))
    assert Leapfrog_QPQ == (("Q", 0.5), ("P", 1.0), ("Q", 0.5))
    for scheme, ops in ((Omelyan, "PQPQP"), (Omelyan_QPQ, "QPQPQ")):
        assert "".join(op for op, _ in scheme) == ops
        assert np.allclose([c for _, c in scheme], [LAM, 0.5, 1 - 2 * LAM, 0.5, LAM])


def test_a_level_takes_any_valid_scheme():
    lam = 0.21
    tuned = (("P", lam), ("Q", 0.5), ("P", 1 - 2 * lam), ("Q", 0.5), ("P", lam))
    assert Level(tuned, Nstep=3).scheme == tuned


@pytest.mark.parametrize(
    "scheme, match",
    [
        ((("P", 0.3), ("Q", 1.0), ("P", 0.7)), "symmetric"),
        ((("P", 0.5), ("Q", 0.9), ("P", 0.5)), "sum"),
        ((("P", 0.25), ("Q", 1.0), ("P", 0.25)), "sum"),
        ((("K", 0.5), ("Q", 1.0), ("K", 0.5)), "'P' or 'Q'"),
    ],
    ids=["asymmetric", "Q sum", "P sum", "unknown op"],
)
def test_a_level_rejects_a_bad_scheme(scheme, match):
    with pytest.raises(ValueError, match=match):
        Level(scheme, Nstep=4)


def test_Nstep_is_static_and_takes_any_integer():
    # A numpy integer (from looping over np.array([...])) is converted, not
    # stored as an array; a non-integer is refused rather than truncated.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        level = Level(Leapfrog, Nstep=np.int64(5))
    assert type(level.Nstep) is int

    with pytest.raises(TypeError):
        Level(Leapfrog, Nstep=2.5)


def test_tau_is_the_only_leaf():
    integrator = _integrator(Omelyan, 4, [(Leapfrog, 2)], tau=0.7)
    leaves = jax.tree.leaves(integrator)

    assert len(leaves) == 1
    assert float(leaves[0]) == pytest.approx(0.7)


def test_shorthands_are_one_velocity_first_level():
    for shorthand, scheme in ((LeapfrogIntegrator, Leapfrog), (OmelyanIntegrator, Omelyan)):
        integrator = shorthand(tau=1.0, Nstep=7)
        assert len(integrator.levels) == 1
        assert integrator.levels[0].scheme == scheme
        assert integrator.levels[0].Nstep == 7
        assert integrator.levels[0].terms is None


# --- One level ----------------------------------------------------------------


@pytest.mark.parametrize("Nstep", [1, 3])
def test_leapfrog_on_a_harmonic_oscillator(Nstep):
    # The textbook one-step maps of both leapfrog orderings.
    h = 0.3
    w2, x0, p0 = W2.F, X0["x"].F, P0["x"].F
    a = 1 - h**2 * w2 / 2

    def pqp(x, p):
        return a * x + h * p, -h * w2 * (1 - h**2 * w2 / 4) * x + a * p

    def qpq(x, p):
        return a * x + h * (1 - h**2 * w2 / 4) * p, -h * w2 * x + a * p

    for scheme, step in ((Leapfrog, pqp), (Leapfrog_QPQ, qpq)):
        x, p = x0, p0
        for _ in range(Nstep):
            x, p = step(x, p)

        integrator = Integrator(tau=h * Nstep, levels=[Level(scheme, Nstep)])
        X, P, _ = integrator.integrate(drift, (harmonic,), X0, P0)
        assert jnp.allclose(X["x"].F, x, atol=1e-13)
        assert jnp.allclose(P["x"].F, p, atol=1e-13)


@pytest.mark.parametrize("name", ["Omelyan", "Omelyan_QPQ"])
def test_omelyan_on_a_harmonic_oscillator(name):
    # Applied step by step: a kick p -= c h w2 x, a drift x += c h p.
    scheme, h, Nstep = SCHEMES[name], 0.3, 3
    w2, x, p = W2.F, X0["x"].F, P0["x"].F
    for _ in range(Nstep):
        for op, c in scheme:
            if op == "P":
                p = p - c * h * w2 * x
            else:
                x = x + c * h * p

    integrator = Integrator(tau=h * Nstep, levels=[Level(scheme, Nstep)])
    X, P, _ = integrator.integrate(drift, (harmonic,), X0, P0)
    assert jnp.allclose(X["x"].F, x, atol=1e-13)
    assert jnp.allclose(P["x"].F, p, atol=1e-13)


# --- Every scheme and nesting -------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_integration_is_reversible(case):
    outer, inner, forces = CASES[case]
    integrator = _integrator(outer, 5, inner)

    X1, P1, _ = integrator.integrate(drift, forces, X0, P0)
    X2, P2, _ = integrator.integrate(drift, forces, X1, {"x": -P1["x"]})

    assert jnp.allclose(X2["x"].F, X0["x"].F, atol=1e-12)
    assert jnp.allclose(-P2["x"].F, P0["x"].F, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_integration_is_second_order(case):
    # Halving every step size cuts the error at fixed tau by 4.
    outer, inner, forces = CASES[case]
    exact, _, _ = _integrator(Omelyan, 4000).integrate(drift, (full,), X0, P0)

    def error(N):
        X, _, _ = _integrator(outer, N, inner).integrate(drift, forces, X0, P0)
        return float(jnp.max(jnp.abs(X["x"].F - exact["x"].F)))

    assert 3.5 < error(8) / error(16) < 4.5


@pytest.mark.parametrize("outer, n_runs", [(Leapfrog, 1), (Leapfrog_QPQ, 2), (Omelyan, 2)])
@pytest.mark.parametrize("inner", [Leapfrog, Omelyan_QPQ], ids=["Leapfrog", "Omelyan_QPQ"])
def test_with_no_outer_force_nesting_is_the_inner_scheme(outer, n_runs, inner):
    # The outer scheme runs the inner one n_runs times per step, over equal
    # intervals: N0 N1 n_runs inner steps in all.
    nested = Integrator(tau=0.8, levels=[Level(outer, 3), Level(inner, 2)])
    flat = Integrator(tau=0.8, levels=[Level(inner, 3 * 2 * n_runs)])

    Xn, Pn, _ = nested.integrate(drift, (zero, full), X0, P0)
    Xf, Pf, _ = flat.integrate(drift, (full,), X0, P0)

    assert jnp.allclose(Xn["x"].F, Xf["x"].F, atol=1e-12)
    assert jnp.allclose(Pn["x"].F, Pf["x"].F, atol=1e-12)


@pytest.mark.parametrize("outer", SCHEMES)
@pytest.mark.parametrize("inner", [Leapfrog, Omelyan_QPQ], ids=["Leapfrog", "Omelyan_QPQ"])
def test_with_no_inner_force_nesting_is_the_outer_scheme(outer, inner):
    # A level with no force is an exact drift, however it is split up.
    nested = Integrator(tau=0.8, levels=[Level(SCHEMES[outer], 3), Level(inner, 2)])
    flat = Integrator(tau=0.8, levels=[Level(SCHEMES[outer], 3)])

    Xn, Pn, _ = nested.integrate(drift, (full, zero), X0, P0)
    Xf, Pf, _ = flat.integrate(drift, (full,), X0, P0)

    assert jnp.allclose(Xn["x"].F, Xf["x"].F, atol=1e-12)
    assert jnp.allclose(Pn["x"].F, Pf["x"].F, atol=1e-12)


# --- Force evaluations --------------------------------------------------------


def _recording(name, force):
    # A force that reports where it was evaluated.
    def recorded(X, P):
        dP, _ = force(X, P)
        return dP, {name: {"x": X["x"].F[0, 0]}}

    return recorded


@pytest.mark.parametrize(
    "name, count",
    [("Leapfrog", lambda N: N + 1), ("Leapfrog_QPQ", lambda N: N),
     ("Omelyan", lambda N: 2 * N + 1), ("Omelyan_QPQ", lambda N: 2 * N)],
)
def test_force_evaluations_of_one_level(name, count):
    integrator = Integrator(tau=1.0, levels=[Level(SCHEMES[name], 4)])
    _, _, info = integrator.integrate(drift, (_recording("f", full),), X0, P0)

    assert info["f"]["x"].shape == (count(4),)


@pytest.mark.parametrize(
    "outer, outer_count, n_runs",
    [(Leapfrog, lambda N: N + 1, 1), (Leapfrog_QPQ, lambda N: N, 2),
     (Omelyan, lambda N: 2 * N + 1, 2)],
    ids=["Leapfrog", "Leapfrog_QPQ", "Omelyan"],
)
def test_force_evaluations_when_nested(outer, outer_count, n_runs):
    # The inner Leapfrog reuses its force across runs, as well as within one:
    # one evaluation to start, then one per inner step.
    N0, N1 = 3, 2
    integrator = Integrator(tau=1.0, levels=[Level(outer, N0), Level(Leapfrog, N1)])
    forces = (_recording("outer", quartic), _recording("inner", harmonic))
    _, _, info = integrator.integrate(drift, forces, X0, P0)

    assert info["outer"]["x"].shape == (outer_count(N0),)
    assert info["inner"]["x"].shape == (1 + n_runs * N0 * N1,)


def test_force_evaluations_are_in_time_order():
    integrator = _integrator(Omelyan, 3, [(Leapfrog, 2)])
    forces = (_recording("outer", quartic), _recording("inner", harmonic))
    X, _, info = integrator.integrate(drift, forces, X0, P0)
    outer, inner = np.asarray(info["outer"]["x"]), np.asarray(info["inner"]["x"])

    # Velocity-first: the first evaluation is at the start, the last at the end.
    for x in (outer, inner):
        assert x[0] == pytest.approx(float(X0["x"].F[0, 0]))
        assert x[-1] == pytest.approx(float(X["x"].F[0, 0]))
    # No evaluation is repeated at a configuration the last one already saw.
    assert np.all(np.diff(inner) != 0)
    # Every outer kick falls between two inner runs, where the inner force was
    # just evaluated: the outer evaluations appear among the inner ones, in order.
    where = [np.flatnonzero(inner == x) for x in outer]
    assert all(len(w) == 1 for w in where)
    assert np.all(np.diff([w[0] for w in where]) > 0)


# --- HMC ----------------------------------------------------------------------


class Quartic(Term):
    lamb: float

    def total(self, phi):
        s = jnp.sum(phi.F**4)
        return self.lamb * s, {"norm4": s}


class Quadratic(Term):
    m2: float

    def total(self, phi):
        s = jnp.sum(phi.F**2)
        return 0.5 * self.m2 * s, {"norm2": s}


ACTION = Action(Quartic(lamb=0.5), Quadratic(m2=1.0))


def _phi(key):
    return {"phi": _field(key)}


def _two_level_hmc(action=ACTION, N0=3, N1=2, tau=1.0):
    return HMC(
        action=action,
        integrator=Integrator(
            tau=tau,
            levels=[
                Level(Omelyan, N0, terms=("Quartic",)),
                Level(Leapfrog, N1, terms=["Quadratic"]),  # any sequence of names
            ],
        ),
    )


def test_hmc_applies_each_term_on_its_own_level():
    # Each term's solves count its own level's force evaluations, plus S at
    # the start and the end of the trajectory.
    _, monitor, _ = _two_level_hmc().evolve(_phi(3), jax.random.PRNGKey(4))

    assert monitor["Quartic.norm4"].shape == (2 * 3 + 1 + 2,)
    assert monitor["Quadratic.norm2"].shape == (1 + 2 * 3 * 2 + 2,)


def test_a_lone_level_applies_the_whole_action():
    hmc = HMC(action=ACTION, integrator=OmelyanIntegrator(tau=1.0, Nstep=3))
    _, monitor, _ = hmc.evolve(_phi(3), jax.random.PRNGKey(4))

    assert monitor["Quartic.norm4"].shape == (2 * 3 + 1 + 2,)
    assert monitor["Quadratic.norm2"].shape == (2 * 3 + 1 + 2,)


@pytest.mark.parametrize(
    "outer_terms, inner_terms, match",
    [
        (("Quartic",), (), "Quadratic"),
        (("Quartic",), ("Quartic", "Quadratic"), "Quartic"),
        (("Quartic", "Nope"), ("Quadratic",), "Nope"),
        (None, ("Quadratic",), "terms"),
    ],
    ids=["missing", "twice", "unknown", "unnamed"],
)
def test_hmc_requires_each_term_on_exactly_one_level(outer_terms, inner_terms, match):
    integrator = Integrator(
        tau=1.0,
        levels=[Level(Omelyan, 3, terms=outer_terms), Level(Leapfrog, 2, terms=inner_terms)],
    )
    with pytest.raises(ValueError, match=match):
        HMC(action=ACTION, integrator=integrator)


def test_nested_hmc_conserves_H_to_second_order(phi4):
    # The check that each level's force is the right one: a term applied on
    # the wrong level, twice, or not at all leaves delta_H of order one.
    action = ScalarTerm(kappa=0.18, lamb=1.145) + Quartic(lamb=0.2)
    fields = {"phi": phi4}

    def mean_abs_dH(N0):
        hmc = HMC(
            action=action,
            integrator=Integrator(
                tau=1.0,
                levels=[Level(Omelyan, N0, terms=("Quartic",)), Level(Leapfrog, 3, terms=("ScalarTerm",))],
            ),
        )
        dHs = []
        for seed in range(4):
            pi, _ = hmc.momentum_refresh(fields, jax.random.PRNGKey(seed))
            dHs.append(abs(float(hmc.MD_traj(fields, pi)[2])))
        return np.mean(dHs)

    assert 3.0 < mean_abs_dH(8) / mean_abs_dH(16) < 5.0


def test_nested_hmc_has_unit_exp_minus_delta_H():
    # Metropolis requires <exp(-delta_H)> = 1, as in test_scalar_hmc.
    L = lat.SquareLattice(st_dims=(4, 4, 4))
    phi = lat.LatticeField(lattice=L, F=jnp.ones((4, 4, 4)))
    action = ScalarTerm(kappa=0.18, lamb=1.145) + Quartic(lamb=0.1)
    hmc = HMC(
        action=action,
        integrator=Integrator(
            tau=1.0,
            levels=[Level(Omelyan, 4, terms=("Quartic",)), Level(Leapfrog, 3, terms=("ScalarTerm",))],
        ),
    )

    fields, key = {"phi": phi}, jax.random.PRNGKey(2024)
    fields, _, key = hmc.as_warmup().evolve_many(fields, key, traj=100)
    _, monitor, _ = hmc.evolve_many(fields, key, traj=500)

    assert abs(float(jnp.mean(jnp.exp(-monitor["delta_H"]))) - 1.0) < 0.1
