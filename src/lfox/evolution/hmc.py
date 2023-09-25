from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import inspect

class Evolver(ABC):
    # Any (MCMC) evolver needs:
    # - An action object, which contains the action functional, the set of fields
    #   to be evolved, and other key auxiliary information
    # - A dictionary of observables to measure, along with measurement frequencies,
    #   in the form:  { 'obs_name': [obs_function, 10] }
    # - An RNG seed.  Note that this always shows the INITIAL seed value used;
    #   the evolving JAX RNG key is saved in the `rng_key` property.

    # Any other hyperparameters for evolution (integrators etc.) should be properties
    # of concrete implementations which inherit from Evolver.
    
    # Properties it should have:
    # - A Markov chain (before evolution, just the initial
    # field configuration lives here)
    # - A dict of lists of observable measurements (paired with configuration #s)

    # Should be general enough to accommodate: 
    # - HMC
    # - Heatbath/OR
    # - Cluster algorithms

    def __init__(self, action, seed, observables=None):
        self.action = action
        self.seed = seed
        self.rng_key = jax.random.PRNGKey(self.seed)

        self.observables = observables

        self.N_fields = len(self.action.fields)

    @abstractmethod
    def evolve(self):
        pass



class Action():
    # An action needs the following to be created:
    # - A dictionary of lattice fields
    # - [optional] A dictionary of non-field parameters the action depends on (e.g. couplings)

    # The action functional _S should depend on the fields and parameters, and
    # should be implemented by any inheriting subclass.

    # Actions can be created by combining two actions together using += or +.
    # This uses the sub_actions parameter.

    def __init__(self, fields, params=None):
        self.fields = fields

        if params is None:
            self.params = {}
        else:
            self.params = params

        self.sub_actions = []

    # Action functional
    def S(self):
        S_tot = 0.0

        # Evaluate main action functional
        S_tot += self._S()

        # Add action functionals for any subclasses
        for action in self.sub_actions:
            S_tot += action._S()

        return S_tot

    def _S(self):
        # This method should be overloaded by concrete actions!
        return 0.0

    # Overload addition with composition
    def __iadd__(self, other):
        self.add_subaction(other)
        return None
    
    def __add__(self, other):
        newAct = self.__copy__()
        newAct.add_subaction(other)

        return newAct
    
    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        for fname in self.fields.keys():
            new.fields[fname] = self.fields[fname].copy()
        
        return new

    def copy(self):
        return self.__copy__()

    def add_subaction(self, other):
        # Combine the fields; in case of name collision, make sure they are really the same field!
        for field_name in other.fields.keys():
            if self.fields.get(field_name) is not None:
                assert self.fields[field_name] is other.fields[field_name]
            
            self.fields[field_name] = other.fields[field_name]
        
        # Combine the parameters
        self.params.update(other.params)

        # Register the subaction
        self.sub_actions.append(other)




class LeapfrogIntegrator():

    def __init__(self, eps, Nstep):
        self.eps = eps
        self.Nstep = Nstep
        
        self.traj_length = eps * Nstep

    @staticmethod
    def update(fields, delta, dt):
        for fname in fields.keys():
            fields[fname] += dt * delta[fname]

    @partial(jax.jit, static_argnums=(0,1,2))
    def integrate(self, delta_X, delta_P, X, P):
        # Note that X and P should both be dictionaries
        # of fields (like in Action()) with matching keys.
        # All fields are modified in place.

        self.update(X, delta_X(X,P), self.eps/2.)
        self.update(P, delta_P(X,P), self.eps)

        for _ in range(self.Nstep-1):
            self.update(X, delta_X(X,P), self.eps)
            self.update(P, delta_P(X,P), self.eps)
        
        self.update(X, delta_X(X,P), self.eps/2.)

        return



class HMCEvolver(Evolver):

    def __init__(self, action, seed, integrator, observables=None):
        self.integrator = integrator
        self.monitor = {
            'delta_H': [],
            'P_acc': [],
        }

        self.field_names = list(action.fields.keys())
        self.field_chain = { fname: [] for fname in self.field_names }


        # No need to initialize momentum fields yet - will happen
        # when the evolution starts
        self.pi_fields = None

        super().__init__(action=action, seed=seed, observables=observables)

    def H(self):
        KE_sum = 0.0
        for fname in self.field_names:
            field = self.pi_fields[fname]
            KE_sum += jnp.sum(field**2)

        return 0.5 * KE_sum + self.action.S()

    def mom_refresh(self):
        if self.pi_fields is None:
            self.pi_fields = {}

        for fname in self.field_names:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.pi_fields[fname] = jax.random.normal(subkey, shape=self.action.fields[fname].field.shape)
    
    # Produce an MD integrator-compatible function
    def delta_mom(self):
        F = self.action.forces
        def delta_P(X, P):
            result = {}
            for field in self.field_names:
                result[field] = F(X)[field]
        
        return delta_P

    def delta_fields(self):
        def delta_X(X, P):
            return P
    
        return delta_X

    def evolve(self):
        # Heatbath momentum refresh
        self.mom_refresh()

        # Store old field values
        prev_action = self.action.copy()
        H_old = self.H()

        # Integrate the trajectory
        self.integrator.integrate(
            delta_X = self.delta_fields(),
            delta_P = self.delta_mom(),
            X = self.action.fields,
            P = self.pi_fields,
        )

        # Accept/reject
        H_new = self.H()
        delta_H = H_new - H_old
        P_acc = jnp.exp(-delta_H)

        self.monitor['delta_H'].append(delta_H)
        self.monitor['P_acc'].append(P_acc)

        if P_acc < 1:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            r = jax.random.uniform(subkey)
            if r > P_acc:
                self.action.fields = prev_action.fields
        
        for fname in self.field_names:
            self.field_chain[fname].append(self.action.fields[fname])

        # Measure observables (TODO)




        

    
