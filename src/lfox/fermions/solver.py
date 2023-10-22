import jax
import jax.numpy as jnp
from functools import partial

class MRSolver():

    def __init__(self, dirac_op, target_resid, omega_OR=1.1, max_iter=1000):
        self.dirac_op = dirac_op
        self.target_resid = target_resid
        self.omega_OR = omega_OR
        self.max_iter = max_iter

    def solve(self, chi, psi_0):
        r = chi - self.dirac_op.op(psi_0)
        psi = psi_0.copy()
        iter = 0
        resid = r.norm_sq()

        while (iter < self.max_iter) and (resid > self.target_resid):
            psi, r = self._MR_step(psi, r)
            resid = r.norm_sq()

        return (psi, resid, iter)

    @partial(jax.jit, static_argnums=(0,))
    def _MR_step(self, psi, r):
        p = self.dirac_op.op(r)
        p_norm = p.norm_sq()
        alpha = self.omega_OR * p.dot(r) / p_norm

        psi_new = psi + alpha * r
        r_new = r - alpha * p

        return (psi_new, r_new)