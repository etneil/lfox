from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax


class MDIntegrator(eqx.Module):
    eps: float
    Nstep: int

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

        X = self.update(X, delta_X(X, P), self.eps / 2.0)
        P = self.update(P, delta_P(X, P), self.eps)

        def mid_step(i, XP):
            Xint, Pint = XP
            Xint = self.update(Xint, delta_X(Xint, Pint), self.eps)
            Pint = self.update(Pint, delta_P(Xint, Pint), self.eps)

            return (Xint, Pint)

        X, P = jax.lax.fori_loop(0, self.Nstep - 1, mid_step, (X, P))

        X = self.update(X, delta_X(X, P), self.eps / 2.0)

        return X, P


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

        def mid_step(i, XP):
            Xint, Pint = XP

            Xint = self.update(Xint, delta_X(Xint, Pint), 2 * xiEps)
            Pint = self.update(Pint, delta_P(Xint, Pint), self.eps / 2.0)
            Xint = self.update(Xint, delta_X(Xint, Pint), middle_step)
            Pint = self.update(Pint, delta_P(Xint, Pint), self.eps / 2.0)

            return (Xint, Pint)

        X = self.update(X, delta_X(X, P), xiEps)
        P = self.update(P, delta_P(X, P), self.eps / 2.0)
        X = self.update(X, delta_X(X, P), middle_step)
        P = self.update(P, delta_P(X, P), self.eps / 2.0)

        X, P = jax.lax.fori_loop(0, self.Nstep - 1, mid_step, (X, P))

        X = self.update(X, delta_X(X, P), xiEps)

        return X, P
