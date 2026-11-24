import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from lfox.action import Action
from lfox.evolution.base import Evolver
from lfox.evolution.integrators import MDIntegrator


class HMC(Evolver):
    # Hybrid Monte Carlo.  Refresh momenta from a Gaussian heatbath, integrate
    # Hamilton's equations along a molecular-dynamics trajectory, then accept or
    # reject on exp(-delta_H).
    #
    # `warmup` is a static field rather than a call argument: "always accept" is a
    # different transition kernel, not a flag on this one.  Build a warmup variant
    # with `as_warmup()`.
    #
    # The monitor reports delta_H, P_acc and accept, plus the solver info of
    # every action evaluation in the trajectory, under flat keys
    # "<term>.<key>": S at the start, each force evaluation, then S at the
    # proposal, stacked in that order.

    action: Action = eqx.field(converter=Action)
    integrator: MDIntegrator
    warmup: bool = eqx.field(static=True, default=False)

    def as_warmup(self):
        return dataclasses.replace(self, warmup=True)

    def H_density(self, S_field, pi_fields):
        # H per site: the action density plus 0.5 p.p, with any internal axes
        # of the momenta contracted.
        H_field = S_field
        for p in pi_fields.values():
            H_field = H_field + 0.5 * p.vdot(p)

        return H_field

    def kinetic(self, pi_fields):
        # Total kinetic energy, sum 0.5 p.p.
        K = 0.0
        for p in pi_fields.values():
            K = K + 0.5 * jnp.sum(p.vdot(p).F)

        return K

    def H(self, S, pi_fields):
        # H from the total action, which every action has; a nonlocal one
        # (a pseudofermion) has no density.  Note that delta_H is then a
        # difference of two O(volume) numbers, which single precision resolves
        # only to ~1e-7 x volume: run the accept/reject in double precision.
        return S + self.kinetic(pi_fields)

    def momentum_refresh(self, fields, rng_key):
        # Gaussian momenta, one per evolved field.  Each is a field like the one
        # it moves -- same lattice, internal axes and boundary conditions -- so
        # that X + eps P is well defined.
        pi_fields = {}
        for fname in self.action.evolved_fields:
            rng_key, subkey = jax.random.split(rng_key)
            field = fields[fname]
            pi_fields[fname] = field.copy_new_F(
                jax.random.normal(subkey, shape=field.F.shape)
            )

        return pi_fields, rng_key

    def delta_fields(self):
        def delta_X(X, P):
            return P

        return delta_X

    def delta_mom(self):
        def delta_P(X, P):
            derivs, info = self.action.dS(X, wrt=tuple(P), info=True)
            return {name: -derivs[name] for name in P}, info

        return delta_P

    def MD_traj(self, fields, pi_fields):
        # Returns the proposal, delta_H, P_acc, and the solver info of the
        # trajectory as flat monitor entries.
        S_old, info_old = self.action.S(fields, info=True)
        H_old = self.H(S_old, pi_fields)

        fields, pi_fields, info_md = self.integrator.integrate(
            delta_X=self.delta_fields(),
            delta_P=self.delta_mom(),
            X=fields,
            P=pi_fields,
        )

        S_new, info_new = self.action.S(fields, info=True)
        H_new = self.H(S_new, pi_fields)

        delta_H = H_new - H_old
        P_acc = jnp.exp(-delta_H)

        # Every solve in order: start, forces, end.
        solves = jax.tree.map(
            lambda a, md, b: jnp.concatenate([a[None], md, b[None]]),
            info_old, info_md, info_new,
        )
        info = {
            f"{term}.{key}": value
            for term, entries in solves.items()
            for key, value in entries.items()
        }

        return fields, pi_fields, delta_H, P_acc, info

    @jax.jit
    def evolve(self, fields, rng_key):

        # Refresh heatbath draws, if any are present
        if len(self.action.sampled_fields) > 0:
            rng_key, subkey = jax.random.split(rng_key)
            fields = fields | self.action.sample(fields, subkey)

        # Refresh momentum.  Note that rng_key is advanced past every momentum
        # draw; the accept/reject uniform must be drawn from the advanced key, or
        # it is a deterministic function of the momenta it is meant to test.
        pi_fields, rng_key = self.momentum_refresh(fields, rng_key)

        # Integrate forward
        new_fields, new_pi_fields, delta_H, P_acc, info = self.MD_traj(fields, pi_fields)

        monitor = {"delta_H": delta_H, "P_acc": P_acc} | info

        if self.warmup:  # Warmups always accept!
            accept = jnp.array(True)
        else:
            rng_key, subkey = jax.random.split(rng_key)
            r = jax.random.uniform(subkey)

            accept = r < P_acc
            new_fields = jax.lax.cond(accept, lambda: new_fields, lambda: fields)

        monitor["accept"] = accept

        return new_fields, monitor, rng_key
