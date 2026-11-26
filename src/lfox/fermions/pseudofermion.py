"""Pseudofermions: two degenerate flavors of any fermion matrix."""

import inspect

import jax
import jax.numpy as jnp

from lfox.action import Term
from lfox.linalg import cg, dagger, stationary_form


class PseudoFermion(Term):
    """S = eta^dag (M^dag M)^-1 eta: two degenerate flavors, one pseudofermion.

    Subclass and supply `fermion_matrix`, the linear map M as a function of
    spinors, closing over the bosonic fields it is called with:

        class YukawaFermion(PseudoFermion):
            kappa: float
            g: float

            def fermion_matrix(self, phi):
                return lambda psi: wilson(psi, self.kappa) + self.g * phi * psi

    The term's roles are the parameters of `fermion_matrix`, then `eta`.  `eta`
    is sampled by the exact heatbath eta = M^dag xi; the bosons are evolved.
    The force is autodiff of `stationary_form`, one solve per evaluation, and
    a subclass may override `gradient` with a hand-written one.  A subclass
    that overrides `total` must override `gradient` as well, since the force
    inherited from here would belong to this action, not to its own.

    More flavors are more pseudofermions: `Action(term).replicate(2, "eta")`
    is four.
    """

    sampled = ("eta",)

    def __init_subclass__(cls, **kwargs):
        if cls.fermion_matrix is not PseudoFermion.fermion_matrix:
            params = list(inspect.signature(cls.fermion_matrix).parameters.values())[1:]
            if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
                raise TypeError(
                    f"{cls.__name__}.fermion_matrix must name its fields "
                    "explicitly: they are the term's roles."
                )
            bosons = tuple(p.name for p in params)
            if "eta" in bosons:
                raise TypeError(
                    f"{cls.__name__}.fermion_matrix may not take 'eta', which is "
                    "the pseudofermion's own role."
                )
            cls.roles = bosons + ("eta",)

        if "total" in cls.__dict__ and "gradient" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} overrides total but not gradient; the inherited "
                "gradient is the force of PseudoFermion's action, not this one's."
            )

        super().__init_subclass__(**kwargs)

    def fermion_matrix(self):
        # Abstract: a subclass names its bosonic fields here and returns M.
        raise NotImplementedError

    def _MdagM(self, eta, bosons):
        # eta is only a template for the space M acts on.
        M = self.fermion_matrix(**bosons)
        M_dag = dagger(M, eta)
        return lambda v: M_dag(M(v))

    def total(self, eta, **bosons):
        x, info = cg(self._MdagM(eta, bosons), eta)
        return jnp.vdot(eta.F, x.F).real, info

    def draw(self, key, eta, **bosons):
        # eta = M^dag xi, with xi of density exp(-xi^dag xi), so that
        # S = xi^dag xi.  The current eta is only a template for the new one.
        k1, k2 = jax.random.split(key)
        shape = eta.F.shape
        xi = (jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)) / jnp.sqrt(2)
        M_dag = dagger(self.fermion_matrix(**bosons), eta)
        return {"eta": M_dag(eta.copy_new_F(xi))}

    def gradient(self, eta, **bosons):
        # dS for every role, by one solve; see `stationary_form`.
        def S(fields):
            bosons = dict(fields)
            eta = bosons.pop("eta")
            return stationary_form(self._MdagM(eta, bosons), eta)

        return jax.grad(S, has_aux=True)({"eta": eta, **bosons})
