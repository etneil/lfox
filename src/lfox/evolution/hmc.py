from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import inspect
import numpy as np
import dataclasses
import copy

import equinox as eqx
from typing import Optional

class Evolver(ABC):
    # Any (MCMC) evolver needs:
    # - An action object, which contains the action functional, the set of fields
    #   to be evolved, and other key auxiliary information
    # - An RNG seed.  Note that this always shows the INITIAL seed value used;
    #   the evolving JAX RNG key is saved in the `rng_key` property.
    # - An initial set of fields, as a "fields" dictionary, keyed on field names
    #   which match self.action.field_names.
    # - (optional) A dictionary of observables to measure, along with measurement frequencies,
    #   in the form:  { 'obs_name': [obs_function, 10] }
    # - (optional) Frequency at which to save field configurations to the internal Markov
    #   chain (default: 1, save every step)

    # Any other hyperparameters for evolution (integrators etc.) should be properties
    # of concrete implementations which inherit from Evolver.
    
    # Properties it should have:
    # - A current field state, as a "fields" dictionary, keyed on field names
    # which match self.action.field_names.
    # - A Markov chain (before evolution, just the initial
    # field configuration lives here)
    # - A dict of lists of observable measurements (paired with configuration #s)

    # Should be general enough to accommodate: 
    # - HMC
    # - Heatbath/OR
    # - Cluster algorithms

    def __init__(self, action, seed, init_fields, observables=None, save_freq=1):
        self.action = action
        self.seed = seed
        self.rng_key = jax.random.PRNGKey(self.seed)
        self.fields = init_fields
        self.save_freq = save_freq

        # Every field should correpond to something in the action
        for key in self.fields.keys():
            assert key in self.action.field_names

        # Every field in the action should be specified
        for key in self.action.field_names:
            assert key in self.fields.keys()

        self.observables = observables

#        self.field_names = list(action.fields.keys())
#        self.field_chain = { fname: [ action.fields[fname] ] for fname in self.field_names }
        self.field_chain = { fname: [init_fields[fname]] for fname in init_fields.keys() }

        if observables is not None:
            self.obs_chain = { obs_name: [] for obs_name in self.observables }

        self.N_fields = len(self.field_chain.keys())


    @abstractmethod
    def evolve(self):
        pass


class EAction(eqx.Module):
    field_names: list
    params: dict

    def __init__(self, field_names, params=None):
        self.field_names = field_names

        if params is None:
            self.params = {}
        else:
            self.params = params

        self.sub_actions = []

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




#class Action(ABC):
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
    params: dict
#    sub_actions: Optional[list] = None
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
        field_call = [ fields[fname] for fname in self.field_names ]

        S_tot = self._S(fields=field_call, params=self.params)

        # Add action functionals for any subclasses
        for action in self.sub_actions:
            field_call = [ fields[fname] for fname in action.field_names ]
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

#    @partial(jax.jit, static_argnums=(0,))
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
        return None
    
    def __add__(self, other):
        newAct = self.__copy__()
        newAct.add_subaction(other)

        return newAct
    
    
    def __copy__(self):
        new_subact = copy.deepcopy(self.sub_actions)
        return dataclasses.replace(self, sub_actions=new_subact)


        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)

        return new

#        new.field_names = self.field_names.copy()

        # Populate sub-action list with copies to avoid
        # unintentional side effects
        new.sub_actions = []
        for subact in self.sub_actions:
            new.sub_actions.append(subact.copy())
                
        return new

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


class MDIntegrator(eqx.Module):
    eps: float
    Nstep: int
        
    def traj_length(self):
        return self.eps * self.Nstep

    @staticmethod
    @jax.jit
    def update(fields, delta, dt):
        new_fields = {}
        for fname in fields.keys():
            new_fields[fname] = fields[fname] + dt * delta[fname]

        return new_fields

    @abstractmethod
    def integrate(self, delta_X, delta_P, X, P):
        pass


class LeapfrogIntegrator(MDIntegrator):

    # Temporary intermediate function for profiling
    def integrate(self, delta_X, delta_P, X, P):
        return self._integrate(delta_X, delta_P, X, P)

    @partial(jax.jit, static_argnums=(1,2))
    def _integrate(self, delta_X, delta_P, X, P):
        # Note that X and P should both be dictionaries
        # of fields (like in Action()) with matching keys.

        X = self.update(X, delta_X(X,P), self.eps/2.)
        P = self.update(P, delta_P(X,P), self.eps)

        def mid_step(i, XP):
            Xint, Pint = XP
            Xint = self.update(Xint, delta_X(Xint,Pint), self.eps)
            Pint = self.update(Pint, delta_P(Xint,Pint), self.eps)

            return (Xint,Pint)
        
        X,P = jax.lax.fori_loop(0, self.Nstep-1, mid_step, (X,P))
        
        X = self.update(X, delta_X(X,P), self.eps/2.)

        return X, P
    

class OmelyanIntegrator(MDIntegrator):
    xi: float = 0.1931833

    # Temporary intermediate function for profiling
    def integrate(self, delta_X, delta_P, X, P):
        return self._integrate(delta_X, delta_P, X, P)

    @partial(jax.jit, static_argnums=(1,2))
    def _integrate(self, delta_X, delta_P, X, P):
        # Note that X and P should both be dictionaries
        # of fields (like in Action()) with matching keys.

        xiEps = self.xi * self.eps
        middle_step = (1 - 2*self.xi) * self.eps

        def mid_step(i, XP):
            Xint,Pint = XP

            Xint = self.update(Xint, delta_X(Xint,Pint), 2*xiEps)
            Pint = self.update(Pint, delta_P(Xint,Pint), self.eps/2.)
            Xint = self.update(Xint, delta_X(Xint,Pint), middle_step)
            Pint = self.update(Pint, delta_P(Xint,Pint), self.eps/2.)

            return (Xint,Pint)

        X = self.update(X, delta_X(X,P), xiEps)
        P = self.update(P, delta_P(X,P), self.eps/2.)
        X = self.update(X, delta_X(X,P), middle_step)
        P = self.update(P, delta_P(X,P), self.eps/2.)

        X,P = jax.lax.fori_loop(0, self.Nstep-1, mid_step, (X,P))
        
        X = self.update(X, delta_X(X,P), xiEps)

        return X, P


from equinox import AbstractVar

class EvolverRewrite(eqx.Module):
    action: Action
    seed: int
    fields: dict
    observables: dict
    save_freq: AbstractVar[int]

    # Should these not be dataclass properties?
    #rng_key: jax.random.PRNGKey

    @abstractmethod
    def evolve(self):
        pass    




class HMCRewrite(EvolverRewrite):
    integrator: MDIntegrator
    save_freq: int = 1

    def evolve(self):
        print("OK")




class HMCEvolver(Evolver):

    def __init__(self, action, seed, init_fields, integrator, 
                 traj_init=0, observables=None, save_freq=1):
        self.integrator = integrator
        self.monitor = {
            'delta_H': [],
            'P_acc': [],
            'accept': [],
        }

        self.traj_init = traj_init      # Initial trajectory number
        self.traj_chain = [ traj_init ]

        super().__init__(action=action, seed=seed, observables=observables, init_fields=init_fields,
                         save_freq=save_freq)

        # Avoid recreating delta functions unnecessarily
        self.make_deltas()

        # Initialize momentum fields
        self.pi_fields = {}
        for fname in self.action.field_names:
            self.pi_fields[fname] = self.fields[fname].copy()


    def H(self):
        return self._H(list(self.pi_fields.values()), self.action.S_field())

    @staticmethod
    @jax.jit
    def _H(pi_fields, S_field):
        H_field = S_field.copy()
        for pi in pi_fields:
            H_field += 0.5 * pi**2

        return jnp.sum(H_field.F)
    
    @staticmethod
    @jax.jit
    def _H_density(pi_fields, S_field):
        H_field = S_field.copy()
        for pi in pi_fields:
            H_field += 0.5 * pi**2

        return H_field

    def mom_refresh(self, ntraj):
        # Refactor to try to speed up a bit...
        for fname in self.action.field_names:
            pi_shape = (ntraj,) + self.fields[fname].F.shape
            self.rng_key, fresh_pi = self._mom_heatbath(pi_shape, self.rng_key)
            self.pi_fields[fname] = fresh_pi


    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _mom_heatbath(shape, key):
        key, subkey = jax.random.split(key)
        return key, jax.random.normal(subkey, shape=shape)
    
    # Produce an MD integrator-compatible function
    def delta_mom(self):
#        F = self.action.get_forces()
        def delta_P(X, P):
            result = {}
            derivs = self.action.dS(X)
            for field in self.action.field_names:
                result[field] = -1 * derivs[field]

            return result
        
        return delta_P

    def delta_fields(self):
        def delta_X(X, P):
            return P
    
        return delta_X

    def make_deltas(self):
        self.delta_P = self.delta_mom()
        self.delta_X = self.delta_fields()

    def reverse(self):
        # Run a reverse trajectory by flipping momenta
        pi_rev = { fname: -1*pi for fname, pi in self.pi_fields.items() }
        field_rev = { fname: F[-1].copy() for fname, F in self.field_chain.items() }

        field_rev, pi_rev = self.integrator.integrate(
            delta_X = self.delta_X,
            delta_P = self.delta_P,
            X = field_rev,
            P = pi_rev,
        )

        return field_rev
    
    def MD_traj(self, fields, pi_fields):
        return self._MD_traj(fields, pi_fields)

        fields, pi_fields, H_new, H_old = self._MD_traj_2(fields, pi_fields)
        delta_H, P_acc = self._compute_AR(H_new, H_old)
        return fields, pi_fields, delta_H, P_acc

    
    @partial(jax.jit, static_argnums=(0,))
    def _MD_traj(self, fields, pi_fields):
        # Static-self JIT probably dangerous, try to test later...
        S_old =  self.action.S_field(fields)
        H_old = self._H_density(list(pi_fields.values()), S_old)

        fields, pi_fields = self.integrator.integrate(
            delta_X = self.delta_X,
            delta_P = self.delta_P,
            X = fields,
            P = pi_fields,
        )

        S_new = self.action.S_field(fields)
        H_new = self._H_density(list(pi_fields.values()), S_new)

        delta_H = jnp.sum(H_new.F - H_old.F)
        P_acc = jnp.exp(-delta_H)

        return fields, pi_fields, delta_H, P_acc
    
    # Trying a different division/JIT scheme
    def _MD_traj_2(self, fields, pi_fields):
        S_old = self.action.S_field(fields)
        H_old = self._H_density(list(pi_fields.values()), S_old)

        fields, pi_fields = self.integrator.integrate(
            delta_X = self.delta_X,
            delta_P = self.delta_P,
            X = fields,
            P = pi_fields,
        )

        S_new = self.action.S_field(fields)
        H_new = self._H_density(list(pi_fields.values()), S_new)

        return fields, pi_fields, H_new, H_old

    @staticmethod
    @jax.jit
    def _compute_AR(H_new, H_old):
        delta_H = jnp.sum(H_new.F - H_old.F)
        P_acc = jnp.exp(-delta_H)

        return delta_H, P_acc

    def get_momentum(self, pi_fields, traj):
        return self._get_momentum(pi_fields, traj)

    @staticmethod
    @jax.jit
    def _get_momentum(pi_fields, traj):
        pi_traj = {}
        for fname in pi_fields.keys():
            pi_traj[fname] = pi_fields[fname].copy()
#            pi_traj[fname].F = pi_traj[fname].F[traj]

        return pi_traj

    def evolve(self, ntraj=1, warmup=False):
        # Record starting trajectory
        start_traj = self.traj_chain[-1]

        # Draw accept/reject numbers for this run
        self.rng_key, subkey = jax.random.split(self.rng_key)
        r_accept = np.array(jax.random.uniform(subkey, shape=(ntraj,)))

        # Heatbath momentum refresh
        self.mom_refresh(ntraj=ntraj)

        for traj in range(1,ntraj+1):

            # Store old field values
#            prev_fields = self.action.copy_fields()
            prev_fields = {fname: field.F for fname, field in self.fields.items() }

            pi_traj = self.get_momentum(self.pi_fields, traj)

            """
            H_old = self.H()

            # Integrate the trajectory
            self.action.fields, self.pi_fields = self.integrator.integrate(
                delta_X = self.delta_X,
                delta_P = self.delta_P,
                X = self.action.fields,
                P = self.pi_fields,
            )

            # Accept/reject
            H_new = self.H()
            delta_H = H_new - H_old
            P_acc = np.exp(-delta_H)

            """
            self.fields, pi_traj, delta_H, P_acc = self.MD_traj(self.fields, pi_traj)

            self.monitor['delta_H'].append(float(delta_H))
            self.monitor['P_acc'].append(float(P_acc))

            accept = True
            if not warmup:  # Warmups always accept!
                if P_acc < 1:
#                    self.rng_key, subkey = jax.random.split(self.rng_key)
#                    r = jax.random.uniform(subkey)
                    if r_accept[traj] > P_acc:
                        accept = False
                        for fname in self.action.field_names:
                            self.fields[fname].F = prev_fields[fname]
#                        self.action.fields = prev_fields

            self.monitor['accept'].append(accept)

            # Record completed trajectory        
            for fname in self.action.field_names:
                # TODO: use save_freq here to modify
                if ( self.save_freq == 1) or (((start_traj + traj) % self.save_freq) == 0):
                    self.field_chain[fname].append(self.fields[fname])
                    self.traj_chain.append(start_traj + traj)

            # Measure observables
            if self.observables is not None:
                for obs in self.observables.keys():
                    obs_f, freq = self.observables[obs]

                    if (freq == 1) or (traj + start_traj) % freq == 0:
                        self.obs_chain[obs].append(obs_f(self.fields, self.action.params))

        





        





        

    
