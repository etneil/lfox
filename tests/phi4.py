import jax
import jax.numpy as jnp
from lfox.action import Action


class ScalarAction(Action):
    """Scalar phi^4 action from the Schaefer reproduction notebook."""

    @staticmethod
    @jax.jit
    def _S(fields, params):
        phi = fields[0]
        S = phi**2
        for ax in range(phi.d()):
            S -= 2 * params['kappa'] * phi * phi.nn_field(ax)
        S += params['lambda'] * (phi**2 - 1)**2
        return S

    @staticmethod
    @jax.jit
    def exact_force(fields, params):
        """Analytic gradient of S; used to validate autodiff."""
        phi = fields[0]
        J = phi.nn_field(axis=0, shift=1) + phi.nn_field(axis=0, shift=-1)
        for ax in range(1, phi.d()):
            J += phi.nn_field(axis=ax, shift=1)
            J += phi.nn_field(axis=ax, shift=-1)
        F = -2 * params['kappa'] * J
        F += 2 * phi.F
        F += 4 * params['lambda'] * (phi.F**2 - 1) * phi.F
        return F
