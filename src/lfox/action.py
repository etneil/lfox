import copy
import dataclasses
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp


class Action(eqx.Module):
    # An action needs the following to be created:
    # - An ordered list of lattice field names
    # - [optional] A dictionary of non-field parameters the action depends on (e.g. couplings)
    # - [optional] A list of "sub_actions", which will be added in whenever S or S_field is computed

    # Although most functions of fields use dictionaries in lfox, we use lists in action definitions.
    # This is to make "aliasing" easy - defining copies of the action with different field names,
    # for example to create a many-flavor fermion action.

    # There is a single abstract method in this class, the action functional _S.  This
    # functional should depend on the fields and the parameters, and must be implemented
    # by any inheriting subclass as a STATIC method.  It should have signature:
    # @staticmethod
    # def _S(fields, params):
    #   (...)
    #
    # where `fields` is a list of fields.  The functions S(self, fields) and S_field(self, fields)
    # take dictionaries of fields, mapping them to lists for _S(fields, params) using
    # Action.field_names.

    # JIT compilation of the action is HIGHLY RECOMMENDED!  This can be done simply
    # by adding the jax.jit decorator, i.e.
    #
    # @staticmethod
    # @jax.jit
    # def _S(fields, params):
    #    (...)
    #
    # (Note that the order of decorators is important, don't swap them!)

    # Actions can be created by combining two actions together using += or +.
    # This uses the sub_actions parameter.

    field_names: list = eqx.field(static=True)
    params: dict = eqx.field(static=True)
    sub_actions: list

    def __init__(self, field_names, params=None, sub_actions=None):
        self.field_names = field_names

        if params is None:
            self.params = {}
        else:
            self.params = params

        if sub_actions is None:
            self.sub_actions = []
        else:
            self.sub_actions = sub_actions

    # Action density functional; returns S as a LatticeField, i.e. not summed.
    @jax.jit
    def S_field(self, fields):
        # Evaluate main action functional
        field_call = [fields[fname] for fname in self.field_names]

        S_tot = self._S(fields=field_call, params=self.params)

        # Add action functionals for any subclasses
        for action in self.sub_actions:
            field_call = [fields[fname] for fname in action.field_names]
            S_tot += action._S(fields=field_call, params=self.params)

        return S_tot

    # Total action
    # Note that we DON'T have to be careful about extra indices here;
    # the action density must already be a scalar per-site, so a simple
    # jnp.sum is guaranteed to be a sum over the lattice sites.
    @jax.jit
    def S(self, fields):
        S_density = self.S_field(fields)
        return jnp.sum(S_density.F)

    @jax.jit
    def dS(self, fields):
        return jax.grad(self.S)(fields)

    @staticmethod
    @abstractmethod
    def _S(fields, params):
        # This method interfaces to a function which can be JIT compiled and which is friendly
        # to computing JAX gradients.
        # Call signature (fields and params) MUST match the names
        # in the dictionaries.
        # This is meant to be maximally flexible; can always be overridden to be more efficient
        # by a subclass.
        pass

    # Overload addition with composition
    def __iadd__(self, other):
        self.add_subaction(other)
        return self

    def __add__(self, other):
        newAct = self.__copy__()
        newAct.add_subaction(other)

        return newAct

    def __copy__(self):
        new_subact = copy.deepcopy(self.sub_actions)
        return dataclasses.replace(self, sub_actions=new_subact)

    def copy(self):
        return self.__copy__()

    def add_subaction(self, other):
        # Combine the parameters
        # TODO: warn in case of collision, which will overwrite...
        self.params.update(other.params)

        # If there are subactions within the other field, promote them up to the current subaction list
        if len(other.sub_actions) > 0:
            for subact in other.sub_actions:
                self.sub_actions.append(subact)

            other.sub_actions = []

        # Register the subaction
        self.sub_actions.append(other)
