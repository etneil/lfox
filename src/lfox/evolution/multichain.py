import warnings
from functools import reduce

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from lfox.evolution.base import Chain


def _stack(trees, what):
    # Stack B pytrees leaf by leaf into one with a leading batch axis.  They
    # must agree in everything but their array values: static fields (Nstep,
    # the lattice, the terms of an action) and array shapes.
    def signature(tree):
        return jax.tree.structure(tree), [jnp.shape(x) for x in jax.tree.leaves(tree)]

    first = signature(trees[0])
    if any(signature(tree) != first for tree in trees[1:]):
        raise ValueError(
            f"all {what} must share one structure (the same static fields, "
            "such as Nstep or the lattice, and the same array shapes) to run "
            "as one batch; run differing members as separate MultiChains."
        )

    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


@eqx.filter_jit
def _evolve_batch(evolver, fields, rng_keys, ntraj):
    def evolve_one(evolver, fields, rng_key):
        return evolver.evolve_many(fields, rng_key, ntraj)

    return jax.vmap(evolve_one)(evolver, fields, rng_keys)


class MultiChain(Chain):
    # A batch of independent Markov chains, run in lockstep as one vmapped
    # computation: a parameter scan, or replica streams at one parameter set.
    #
    #   mc = MultiChain(
    #       evolvers=[HMC(action=Phi4(kappa=k, lamb=1.1689), integrator=...) for k in kappas],
    #       seeds=[100 + i for i in range(len(kappas))],
    #       init_fields={'phi': phi},    # shared; or a list, one dict per member
    #       measure=[mag],               # written for one chain
    #   )
    #   mc.warmup(100)
    #   mc.run(1000)
    #   mc.column('mag')                 # (member, Monte Carlo time)
    #
    # Member i is Chain(evolvers[i], seeds[i], init_fields[i]), to round-off.
    # The members may differ in any traced parameter (a coupling, eps), but
    # not in static structure (Nstep, the lattice).  Everything per member
    # carries a leading batch axis: `monitor[key][i]`, `column(name)[i]`,
    # `acceptance()[i]`, and the stacked `fields` and `field_chain` entries.
    # A stacked field is NOT a field on the lattice it names -- a function
    # reducing over its sites would silently reduce over the wrong axes -- so
    # use `member_fields(i)` to get one member's fields.  Trajectory
    # bookkeeping (`traj`, `traj_chain`, `column_traj`) is shared.
    #
    # Within a batch, every member costs as much as the most expensive one:
    # a solve runs until the slowest member converges.  Members of very
    # different cost belong in separate MultiChains.

    def __init__(self, evolvers, seeds, init_fields, measure=None, save_freq=1):
        evolvers, seeds = list(evolvers), tuple(seeds)
        if not evolvers:
            raise ValueError("a MultiChain needs at least one member.")
        if len(seeds) != len(evolvers):
            raise ValueError(f"{len(evolvers)} evolvers but {len(seeds)} seeds.")
        if isinstance(init_fields, dict):
            init_fields = [init_fields] * len(evolvers)
        if len(init_fields) != len(evolvers):
            raise ValueError(
                f"{len(evolvers)} evolvers but {len(init_fields)} init_fields."
            )

        # Allowed, since it can be deliberate, but almost always a mistake.
        repeated = sorted({s for s in seeds if seeds.count(s) > 1})
        if repeated:
            warnings.warn(
                f"seeds {repeated} are used by more than one member.", stacklevel=2
            )

        self._member = dict(init_fields[0]), evolvers[0].action
        super().__init__(
            evolver=_stack(evolvers, "evolvers"),
            seed=seeds,
            init_fields=_stack([dict(f) for f in init_fields], "init_fields"),
            measure=measure,
            save_freq=save_freq,
        )

    def __len__(self):
        return len(self.seed)

    def member_fields(self, i):
        # Member i's current fields, as ordinary LatticeFields.  Indexed like
        # a Python sequence: a JAX array would clamp an index past the end.
        i = range(len(self))[i]

        return jax.tree.map(lambda x: x[i], self.fields)

    def _one_member(self):
        return self._member

    def _rng_key(self, seeds):
        return jax.vmap(jax.random.PRNGKey)(jnp.array(seeds))

    def _compile(self, fn):
        return jax.jit(jax.vmap(fn))

    def _evolve_many(self, evolver, ntraj):
        return _evolve_batch(evolver, self.fields, self.rng_key, ntraj)

    def _record_monitor(self, monitor):
        for key, vals in monitor.items():
            rows = self.monitor.setdefault(key, [[] for _ in range(len(self))])
            for row, member_vals in zip(rows, np.asarray(vals).tolist()):
                row.extend(member_vals)

        for key, vals in monitor.items():
            if not key.endswith(".converged"):
                continue
            converged = np.asarray(vals).reshape(len(self), -1).all(axis=1)
            if not converged.all():
                warnings.warn(
                    f"solves of {key.removesuffix('.converged')!r} did not "
                    f"converge in this block for members "
                    f"{np.flatnonzero(~converged).tolist()}; see the "
                    "'<term>.converged' entries of the monitor.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def acceptance(self):
        # Mean acceptance of each member over production trajectories.
        if "accept" not in self.monitor or self.traj == 0:
            return None

        return np.mean(np.asarray(self.monitor["accept"])[:, -self.traj :], axis=1)

    def column(self, name):
        # Every recorded value of one measurement, as a float64 array of shape
        # (member, Monte Carlo time, ...).
        self._check_name(name)
        values = np.array(self.measurements[name], dtype=np.float64)
        values = values.reshape(-1, len(self), *self.column_shapes[name])

        return np.moveaxis(values, 0, 1)

    def columns(self, *names):
        trajs = [self.column_traj(name) for name in names]
        shared = reduce(np.intersect1d, trajs)

        return tuple(
            self.column(name)[:, np.isin(traj, shared)]
            for name, traj in zip(names, trajs)
        )
