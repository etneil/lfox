import operator
from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp


class MDIntegrator(eqx.Module):
    # Integrates Hamilton's equations, given delta_X(X, P) -> dX and
    # delta_P(X, P) -> (dP, info).  `integrate` returns (X, P, info), with the
    # info of every force evaluation stacked in order along a leading axis.
    #
    # Nstep is static: the number of force evaluations fixes the shape of the
    # stacked info, so changing it recompiles.  Any integer is accepted (a
    # numpy one is converted, as a static array would be a mistake); a
    # non-integer is refused rather than truncated.
    eps: float
    Nstep: int = eqx.field(static=True, converter=operator.index)

    def traj_length(self):
        return self.eps * self.Nstep

    @staticmethod
    @jax.jit
    def update(fields, delta, dt):
        new_fields = dict(fields)
        for fname in delta:
            new_fields[fname] = fields[fname] + dt * delta[fname]

        return new_fields

    @abstractmethod
    def integrate(self, delta_X, delta_P, X, P):
        pass


class LeapfrogIntegrator(MDIntegrator):

    # Temporary intermediate function for profiling
    def integrate(self, delta_X, delta_P, X, P):
        return self._integrate(delta_X, delta_P, X, P)

    @partial(jax.jit, static_argnums=(1, 2))
    def _integrate(self, delta_X, delta_P, X, P):
        # Note that X and P should both be dictionaries
        # of fields (like in Action()) with matching keys.

        # Nstep drift-kick-drift steps, one force evaluation each.  The
        # half-drifts of neighboring steps add up to the usual leapfrog.
        def step(XP, _):
            X, P = XP
            X = self.update(X, delta_X(X, P), self.eps / 2.0)
            dP, info = delta_P(X, P)
            P = self.update(P, dP, self.eps)
            X = self.update(X, delta_X(X, P), self.eps / 2.0)

            return (X, P), info

        (X, P), info = jax.lax.scan(step, (X, P), length=self.Nstep)

        return X, P, info


class OmelyanIntegrator(MDIntegrator):
    xi: float = 0.1931833

    # Temporary intermediate function for profiling
    def integrate(self, delta_X, delta_P, X, P):
        return self._integrate(delta_X, delta_P, X, P)

    @partial(jax.jit, static_argnums=(1, 2))
    def _integrate(self, delta_X, delta_P, X, P):
        # Note that X and P should both be dictionaries
        # of fields (like in Action()) with matching keys.

        xiEps = self.xi * self.eps
        middle_step = (1 - 2 * self.xi) * self.eps

        # Nstep steps of drift-kick-drift-kick-drift, two force evaluations each.
        def step(XP, _):
            X, P = XP
            X = self.update(X, delta_X(X, P), xiEps)
            dP, info_a = delta_P(X, P)
            P = self.update(P, dP, self.eps / 2.0)
            X = self.update(X, delta_X(X, P), middle_step)
            dP, info_b = delta_P(X, P)
            P = self.update(P, dP, self.eps / 2.0)
            X = self.update(X, delta_X(X, P), xiEps)

            return (X, P), jax.tree.map(lambda a, b: jnp.stack([a, b]), info_a, info_b)

        (X, P), info = jax.lax.scan(step, (X, P), length=self.Nstep)
        # (Nstep, 2, ...) -> (2 Nstep, ...), in evaluation order.
        info = jax.tree.map(lambda a: a.reshape(-1, *a.shape[2:]), info)

        return X, P, info
