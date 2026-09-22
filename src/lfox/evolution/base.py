import math
from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax
import numpy as np
from equinox import AbstractVar

from lfox.action import Action


class Evolver(eqx.Module):
    # A Markov transition kernel: given a set of fields and an RNG key, produce a
    # new set of fields that leaves the target distribution exp(-S) invariant.
    #
    # Evolvers are PURE.  They carry only what is constant across the chain (the
    # action, and whatever hyperparameters the algorithm needs).  The evolving
    # state -- the fields and the RNG key -- flows through `evolve` as arguments
    # and back out as return values.  This is what makes `evolve` jittable; do
    # not add mutable state to an Evolver subclass.
    #
    # The chain-level bookkeeping that used to live here (seed, observables,
    # field history, save frequency) belongs to `Chain`, which drives an Evolver
    # from the host side.
    #
    # Subclasses implement a single method:
    #
    #   evolve(fields, rng_key) -> (fields, monitor, rng_key)
    #
    # where `monitor` is a dict of per-step diagnostics.  Its keys are up to the
    # algorithm (HMC reports delta_H / P_acc / accept; a cluster update might
    # report cluster size).  Every value must be a JAX array so that
    # `evolve_many` can stack them.
    #
    # This interface is deliberately thin.  HMC, heatbath and cluster algorithms
    # share almost nothing beyond it: HMC needs dS (free, via autodiff), while
    # heatbath needs local conditional distributions and cluster updates need
    # bond weights.  Those requirements belong on the Action, not here.

    action: AbstractVar[Action]

    @abstractmethod
    def evolve(self, fields, rng_key):
        # One Markov step.  Returns (new_fields, monitor, new_rng_key).
        pass

    def as_warmup(self):
        # Return a variant of this kernel suitable for thermalization.  The
        # default is to change nothing; algorithms with an accept/reject step
        # override this to skip it.  A warmup kernel does NOT preserve the
        # target distribution, so never measure on configurations it produced.
        return self

    @partial(jax.jit, static_argnums=(3,), static_argnames=("traj",))
    def evolve_many(self, fields, rng_key, traj=1):
        # Run `traj` steps inside a single lax.scan.  scan stacks the per-step
        # monitor pytree for us, so this works for any monitor layout without
        # the base class needing to know the keys in advance.

        def step(carry, _):
            fields, rng_key = carry
            fields, monitor, rng_key = self.evolve(fields, rng_key)

            return (fields, rng_key), monitor

        (fields, rng_key), monitor = jax.lax.scan(
            step, (fields, rng_key), xs=None, length=traj
        )

        return fields, monitor, rng_key


def _normalize_observables(observables):
    # Accept either {'name': fn} (measured every trajectory) or
    # {'name': [fn, freq]}, and normalize to {'name': (fn, freq)}.
    normalized = {}
    for name, spec in (observables or {}).items():
        if callable(spec):
            normalized[name] = (spec, 1)
        else:
            obs_f, freq = spec
            normalized[name] = (obs_f, int(freq))

    return normalized


class Chain:
    # Host-side driver for an Evolver.  Owns everything the transition kernel
    # deliberately does not: the RNG key, the current fields, the Markov chain of
    # saved configurations, per-step diagnostics, and observable measurements.
    #
    # This is an ordinary Python class, not an eqx.Module -- it is stateful and
    # lives entirely outside any jit boundary.  All the compiled work happens in
    # `evolver.evolve_many`.
    #
    #   chain = Chain(evolver=HMC(...), seed=42, init_fields={'phi': phi})
    #   chain.warmup(100)
    #   chain.run(500)
    #
    # `observables` maps a name to either a function f(fields, params) or a pair
    # [f, freq] measuring every `freq` trajectories.
    #
    # Trajectory numbering counts production trajectories only; `warmup` advances
    # `warmup_traj` instead and saves nothing.

    def __init__(self, evolver, seed, init_fields, observables=None, save_freq=1):
        self.evolver = evolver
        self.seed = seed
        self.rng_key = jax.random.PRNGKey(seed)
        self.fields = dict(init_fields)
        self.save_freq = save_freq
        self.observables = _normalize_observables(observables)

        # Every field must correspond to something in the action, and vice versa
        field_names = set(evolver.action.field_names)
        if set(self.fields) != field_names:
            raise ValueError(
                f"Field names {sorted(self.fields)} do not match the action's "
                f"field names {sorted(field_names)}."
            )

        self.traj = 0
        self.warmup_traj = 0

        self.traj_chain = [0]
        self.field_chain = {fname: [self.fields[fname]] for fname in self.fields}

        self.monitor = {}
        self.obs_chain = {name: [] for name in self.observables}
        self.obs_traj = {name: [] for name in self.observables}

    @property
    def N_fields(self):
        return len(self.fields)

    def _block_size(self):
        # Step in blocks large enough to keep the scan busy, but small enough
        # that we never step over a trajectory at which something must be
        # recorded.  Every save/measurement frequency is a multiple of this.
        block = self.save_freq
        for _, freq in self.observables.values():
            block = math.gcd(block, freq)

        return max(block, 1)

    def _record_monitor(self, monitor):
        for key, vals in monitor.items():
            self.monitor.setdefault(key, []).extend(np.asarray(vals).tolist())

    def _record_step(self):
        if self.traj % self.save_freq == 0:
            for fname in self.fields:
                self.field_chain[fname].append(self.fields[fname])
            self.traj_chain.append(self.traj)

        for name, (obs_f, freq) in self.observables.items():
            if self.traj % freq == 0:
                self.obs_chain[name].append(
                    obs_f(self.fields, self.evolver.action)
                )
                self.obs_traj[name].append(self.traj)

    def warmup(self, ntraj):
        # Thermalize.  Diagnostics are recorded; configurations and observables
        # are not, since a warmup kernel does not sample the target distribution.
        evolver = self.evolver.as_warmup()
        self.fields, monitor, self.rng_key = evolver.evolve_many(
            self.fields, self.rng_key, ntraj
        )
        self.warmup_traj += ntraj
        self._record_monitor(monitor)

        return self

    def run(self, ntraj):
        block = self._block_size()
        if ntraj % block != 0:
            raise ValueError(
                f"ntraj={ntraj} must be a multiple of {block}, the greatest common "
                f"divisor of save_freq and the observable frequencies."
            )

        for _ in range(ntraj // block):
            self.fields, monitor, self.rng_key = self.evolver.evolve_many(
                self.fields, self.rng_key, block
            )
            self.traj += block
            self._record_monitor(monitor)
            self._record_step()

        return self

    def acceptance(self):
        # Mean acceptance over production trajectories, if the kernel reports it.
        if "accept" not in self.monitor or self.traj == 0:
            return None

        return float(np.mean(self.monitor["accept"][-self.traj :]))
