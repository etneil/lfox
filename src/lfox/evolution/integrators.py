import operator
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# A scheme is a symmetric sequence of steps (op, c), applied once per step of
# size h: "P" kicks the momenta with the level's force for time c h, and "Q"
# advances the positions for time c h.  Symmetry makes the step reversible; the
# P and the Q coefficients each sum to 1.  The velocity-first (PQP) forms are
# the defaults.  lambda is lambda_c of Takaishi and de Forcrand
# (hep-lat/0505020); the optimum varies slightly with the theory, its
# parameters and the ordering, so to tune it, write the tuple.
_LAMBDA = 0.1931833

Leapfrog = (("P", 0.5), ("Q", 1.0), ("P", 0.5))
Leapfrog_QPQ = (("Q", 0.5), ("P", 1.0), ("Q", 0.5))
Omelyan = (
    ("P", _LAMBDA), ("Q", 0.5), ("P", 1 - 2 * _LAMBDA), ("Q", 0.5), ("P", _LAMBDA)
)
Omelyan_QPQ = (
    ("Q", _LAMBDA), ("P", 0.5), ("Q", 1 - 2 * _LAMBDA), ("P", 0.5), ("Q", _LAMBDA)
)


def _scheme(scheme):
    scheme = tuple((op, float(c)) for op, c in scheme)
    if any(op not in ("P", "Q") for op, _ in scheme):
        raise ValueError(f"scheme steps must be 'P' or 'Q': {scheme}.")

    ops = [op for op, _ in scheme]
    coeffs = np.array([c for _, c in scheme])
    if ops != ops[::-1] or not np.allclose(coeffs, coeffs[::-1]):
        raise ValueError(f"scheme {scheme} is not symmetric, so not reversible.")

    for kind in "PQ":
        total = sum(c for op, c in scheme if op == kind)
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"the {kind} coefficients of scheme {scheme} sum to {total}, not 1."
            )

    return scheme


def _terms(terms):
    # A bare string is one name, not a sequence of one-letter names.
    if terms is None:
        return None

    return (terms,) if isinstance(terms, str) else tuple(terms)


class Level(eqx.Module):
    # One timescale: a scheme, its number of steps (per trajectory for the
    # outermost level, per enclosing Q step otherwise), and the names of the
    # action terms whose force it applies.  A lone level may leave `terms`
    # out, and then applies the whole action.
    #
    # Everything is static: Nstep fixes the number of force evaluations, and
    # so the shape of the stacked solver info.  A numpy integer is converted;
    # a non-integer is refused rather than truncated.
    scheme: tuple = eqx.field(static=True, converter=_scheme)
    Nstep: int = eqx.field(static=True, converter=operator.index)
    terms: tuple | None = eqx.field(static=True, default=None, converter=_terms)

    @property
    def reuses_force(self):
        # A scheme that begins and ends with P kicks twice at the same
        # configuration, the end of one step and the start of the next.
        return self.scheme[0][0] == "P" and self.scheme[-1][0] == "P"


def _concat(infos):
    # Join info pytrees along their leading (evaluation) axis.
    return jax.tree.map(lambda *xs: jnp.concatenate(xs), *infos)


def _one(info):
    # The info of a single evaluation, with an evaluation axis of length 1.
    return jax.tree.map(lambda x: x[None], info)


class Integrator(eqx.Module):
    # Integrates Hamilton's equations over a trajectory of length `tau`, on
    # one or more nested timescales, outermost first.  Put expensive, smooth
    # forces (a pseudofermion) on outer levels and cheap, stiff ones inside.
    #
    # A Q step of level k runs level k + 1 over its interval; only the
    # innermost level's Q steps move the fields.  So the outer step is
    # h_0 = tau / N_0, and a Q step of coefficient c at level k runs N_{k+1}
    # steps of h_{k+1} = c h_k / N_{k+1}.
    #
    # A level whose scheme begins and ends with P evaluates its force once at
    # the start, and reuses the force from the end of each step at the start
    # of the next -- also across consecutive runs of an inner level, between
    # which only outer kicks happen.  Leapfrog costs N + 1 force evaluations,
    # Omelyan 2N + 1.
    #
    # `tau` is the only pytree leaf, so it is traced; changing a scheme or an
    # Nstep recompiles.
    tau: float
    levels: tuple[Level, ...] = eqx.field(converter=tuple)

    @staticmethod
    def update(fields, delta, dt):
        new_fields = dict(fields)
        for fname in delta:
            new_fields[fname] = fields[fname] + dt * delta[fname]

        return new_fields

    def integrate(self, delta_X, delta_P, X, P):
        # delta_X(X, P) -> dX; delta_P is a tuple of one force per level,
        # outermost first, each delta_P[k](X, P) -> (dP, info).  Returns
        # (X, P, info): each level's info stacked over its force evaluations
        # in time order, and the levels' info dicts merged.
        if len(delta_P) != len(self.levels):
            raise ValueError(
                f"{len(self.levels)} integrator levels, but {len(delta_P)} forces."
            )

        return self._integrate(delta_X, tuple(delta_P), X, P)

    @partial(jax.jit, static_argnums=(1, 2))
    def _integrate(self, delta_X, delta_P, X, P):
        # Levels that reuse their force start from one evaluated here.
        forces, first = [], []
        for force, level in zip(delta_P, self.levels):
            if level.reuses_force:
                dP, info = force(X, P)
                forces.append(dP)
                first.append(_one(info))
            else:
                forces.append(None)
                first.append(None)

        state = (X, P, tuple(forces))
        (X, P, _), infos = self._run(0, self.tau, state, delta_X, delta_P)

        merged = {}
        for start, rest in zip(first, infos):
            merged |= rest if start is None else _concat([start, rest])

        return X, P, merged

    def _run(self, k, dt, state, delta_X, delta_P):
        # Run level k for time dt.  The state is (X, P, forces), with the
        # force each reusing level last evaluated.  Returns the new state and
        # the info of levels k, k + 1, ..., each stacked in time order.
        level = self.levels[k]
        h = dt / level.Nstep
        innermost = k == len(self.levels) - 1
        last = len(level.scheme) - 1

        def step(state, _):
            X, P, forces = state
            infos = [[] for _ in self.levels[k:]]

            for i, (op, c) in enumerate(level.scheme):
                if op == "P":
                    if level.reuses_force and i == 0:
                        dP = forces[k]
                    else:
                        dP, info = delta_P[k](X, P)
                        infos[0].append(_one(info))
                        if level.reuses_force and i == last:
                            forces = forces[:k] + (dP,) + forces[k + 1 :]
                    P = self.update(P, dP, c * h)
                elif innermost:
                    X = self.update(X, delta_X(X, P), c * h)
                else:
                    (X, P, forces), inner = self._run(
                        k + 1, c * h, (X, P, forces), delta_X, delta_P
                    )
                    for j, info in enumerate(inner):
                        infos[j + 1].append(info)

            return (X, P, forces), tuple(_concat(i) for i in infos)

        state, infos = jax.lax.scan(step, state, length=level.Nstep)
        # (Nstep, evaluations per step, ...) -> (evaluations, ...)
        infos = jax.tree.map(lambda a: a.reshape(-1, *a.shape[2:]), infos)

        return state, infos


def LeapfrogIntegrator(tau, Nstep):
    # One velocity-first leapfrog level.
    return Integrator(tau=tau, levels=[Level(Leapfrog, Nstep)])


def OmelyanIntegrator(tau, Nstep):
    # One velocity-first Omelyan level.
    return Integrator(tau=tau, levels=[Level(Omelyan, Nstep)])
