import jax
import jax.numpy as jnp
from lfox.action import Action, Term

class ScalarTerm(Term):
    """O(N) phi^4, in the hopping convention of the Schaefer reproduction notebook.

        S = sum_x [ phi.phi - 2 kappa sum_mu phi_x . phi_{x+mu} + lamb (phi.phi - 1)^2 ]

    Written with `dot`, so it is one term for any N and any axis name; a field
    with no internal axes is the plain scalar, N = 1.
    """

    kappa: float
    lamb: float

    def density(self, phi):
        phi2 = phi.dot(phi)
        S = phi2 + self.lamb * (phi2 - 1) ** 2
        for ax in range(phi.d()):
            S -= 2 * self.kappa * phi.dot(phi.nn_field(ax))
        return S

    def exact_gradient(self, phi):
        """Analytic gradient of S; used to validate autodiff."""
        J = phi.nn_field(axis=0, shift=1) + phi.nn_field(axis=0, shift=-1)
        for ax in range(1, phi.d()):
            J += phi.nn_field(axis=ax, shift=1)
            J += phi.nn_field(axis=ax, shift=-1)
        grad = -2 * self.kappa * J + 2 * phi
        grad += 4 * self.lamb * (phi.dot(phi) - 1) * phi
        return {'phi': grad}
