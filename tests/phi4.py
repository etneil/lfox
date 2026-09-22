import jax
import jax.numpy as jnp
from lfox.action import Action, Term

class ScalarTerm(Term):
    """Scalar phi^4 action from the Schaefer reproduction notebook."""

    kappa: float
    lamb: float

    def density(self, phi):
        S = phi**2
        for ax in range(phi.d()):
            S -= 2 * self.kappa * phi * phi.nn_field(ax)
        S += self.lamb * (phi**2 - 1)**2
        return S

    def exact_gradient(self, phi):
        """Analytic gradient of S; used to validate autodiff."""
        J = phi.nn_field(axis=0, shift=1) + phi.nn_field(axis=0, shift=-1)
        for ax in range(1, phi.d()):
            J += phi.nn_field(axis=ax, shift=1)
            J += phi.nn_field(axis=ax, shift=-1)
        F = -2 * self.kappa * J
        F += 2 * phi.F
        F += 4 * self.lamb * (phi.F**2 - 1) * phi.F
        return {'phi': F}
