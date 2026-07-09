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

    action: Action
    integrator: MDIntegrator
    warmup: bool = eqx.field(static=True, default=False)

    def as_warmup(self):
        return dataclasses.replace(self, warmup=True)

    def H_density(self, S_field, pi_fields):
        H_field = S_field.copy()
        for fname in pi_fields.keys():
            H_field += 0.5 * pi_fields[fname] ** 2

        return H_field

    def H(self, S_field, pi_fields):
        H_field = self.H_density(S_field, pi_fields)

        return jnp.sum(H_field.F)

    def momentum_refresh(self, fields, rng_key):
        pi_fields = {}
        for fname in fields.keys():
            rng_key, subkey = jax.random.split(rng_key)
            pi_fields[fname] = jax.random.normal(subkey, shape=fields[fname].F.shape)

        return pi_fields, rng_key

    def delta_fields(self):
        def delta_X(X, P):
            return P

        return delta_X

    def delta_mom(self):
        def delta_P(X, P):
            result = {}
            derivs = self.action.dS(X)
            for field in self.action.field_names:
                result[field] = -1 * derivs[field]

            return result

        return delta_P

    def MD_traj(self, fields, pi_fields):
        S_old = self.action.S_field(fields)
        H_old = self.H_density(S_old, pi_fields)

        fields, pi_fields = self.integrator.integrate(
            delta_X=self.delta_fields(),
            delta_P=self.delta_mom(),
            X=fields,
            P=pi_fields,
        )

        S_new = self.action.S_field(fields)
        H_new = self.H_density(S_new, pi_fields)

        delta_H = jnp.sum(H_new.F - H_old.F)
        P_acc = jnp.exp(-delta_H)

        return fields, pi_fields, delta_H, P_acc

    @jax.jit
    def evolve(self, fields, rng_key):
        # Refresh momentum
        pi_fields, new_rng_key = self.momentum_refresh(fields, rng_key)

        # Integrate forward
        new_fields, new_pi_fields, delta_H, P_acc = self.MD_traj(fields, pi_fields)

        monitor = {"delta_H": delta_H, "P_acc": P_acc}

        if self.warmup:  # Warmups always accept!
            accept = jnp.array(True)
        else:
            new_rng_key, subkey = jax.random.split(rng_key)
            r = jax.random.uniform(subkey)

            accept = r < P_acc
            new_fields = jax.lax.cond(accept, lambda: new_fields, lambda: fields)

        monitor["accept"] = accept

        return new_fields, monitor, new_rng_key
