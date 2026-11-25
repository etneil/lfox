import jax
import jax.numpy as jnp

from lfox.action import Term
from lfox.fermions import wilson
from lfox.linalg import cg, dagger


class Pseudofermion(Term):
    """Two degenerate flavors of a Yukawa-Wilson fermion, by one pseudofermion.

        M(phi) psi = wilson(psi, kappa) + g phi psi
        S = eta^dag (M^dag M)^-1 eta

    Integrating out eta gives det(M^dag M) = |det M|^2.  Nonlocal, so it has a
    `total` and no density; the force is autodiff through the solve.
    """

    kappa: float
    g: float
    sampled = ("eta",)

    def operator(self, phi):
        return lambda psi: wilson(psi, self.kappa) + self.g * phi * psi

    def total(self, phi, eta):
        M = self.operator(phi)
        M_dag = dagger(M, eta)
        x, info = cg(lambda v: M_dag(M(v)), eta)
        return jnp.vdot(eta.F, x.F).real, info

    def draw(self, key, phi, eta):
        # eta = M^dag xi, with xi of density exp(-xi^dag xi), so that
        # S = xi^dag xi.  The current eta is only a template for the new one.
        k1, k2 = jax.random.split(key)
        shape = eta.F.shape
        xi = (jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)) / jnp.sqrt(2)
        M_dag = dagger(self.operator(phi), eta)
        return {"eta": M_dag(eta.copy_new_F(xi))}
