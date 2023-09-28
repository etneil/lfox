from abc import ABC, abstractmethod
from functools import partial
import jax
import jax.numpy as jnp
import inspect
import numpy as np

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

        self.field_names = list(action.fields.keys())
        self.field_chain = { fname: [ action.fields[fname] ] for fname in self.field_names }

        if observables is not None:
            self.obs_chain = { obs_name: [] for obs_name in self.observables }

        self.N_fields = len(self.action.fields)


    @abstractmethod
    def evolve(self):
        pass



class Action(ABC):
    # An action needs the following to be created:
    # - A dictionary of lattice fields
    # - [optional] A dictionary of non-field parameters the action depends on (e.g. couplings)

    # The action functional _S should depend on the fields and parameters, and
    # should be implemented by any inheriting subclass.

    # Actions can be created by combining two actions together using += or +.
    # This uses the sub_actions parameter.

    # TODO: I should decouple the action itself from the fields.  The action needs to know the NAMES
    # of the fields, but it's mainly providing function calls to be used by JAX that I want to use
    # within JIT compiled code, which means passing actual numerical fields as function arguments,
    # not having them attached to an Action object.

    def __init__(self, fields, params=None):
        self.fields = fields

        if params is None:
            self.params = {}
        else:
            self.params = params

        self.sub_actions = []
        self.forces = None

    # Action functional
    def S(self):
        # Evaluate main action functional
        S_tot = self._S()

        # Add action functionals for any subclasses
        for action in self.sub_actions:
            S_tot += action._S()

        return S_tot

    def get_forces(self, recompute=False):
        if recompute or self.forces == None:
            self._compute_forces()

        return self.forces

    @staticmethod
    def _safe_call(F, dicts):
        F_sig = inspect.signature(F)

        call = {}
        for D in dicts:
            for k in D.keys():
                if k in F_sig.parameters:
                    call[k] = D[k]
        return F(**call)

    def _compute_forces(self):
        action_sig = inspect.signature(self._Sjax)
        action_pars = list(action_sig.parameters)

        grads = {}
        for fname in self.fields.keys():
            grads[fname] = []

            # Compute gradient of total action with respect to each field
            # Start with the main action
            field_i = action_pars.index(fname)
            grads[fname].append(jax.grad(self._Sjax, argnums=field_i))

            for subact in self.sub_actions:
                subact_pars = list(inspect.signature(subact._Sjax).parameters)
                field_i = subact_pars.index(fname)
                grads[fname].append(jax.grad(subact._Sjax, argnums=field_i))

        # Combine into a single function that returns a dictionary matching self.fields
        def force_func(fields):
            forces = {}
            for fname in fields.keys():
                total_force = []
                for G in grads[fname]:
                    total_force.append(self._safe_call(G, (fields, self.params)))

                forces[fname] = sum(total_force)

            return forces

        self.forces = force_func


    def _S(self):
        # This method interfaces to a function which can be JIT compiled and which is friendly
        # to computing JAX gradients.
        # Call signature (fields and params) MUST match the names
        # in the dictionaries.
        # This is meant to be maximally flexible; can always be overridden to be more efficient
        # by a subclass.


        return self._safe_call(self._Sjax, (self.fields, self.params))
    
    def _S_fields(self, fields):
        # Exposes the fields instead of using the action object

        return self._safe_call(self._Sjax, (fields, self.params))

        action_sig = inspect.signature(self._Sjax)
    
        call = {}
        for k in self.fields.keys():
            if k in action_sig.parameters:
                call[k] = self.fields[k]
        for k in self.params.keys():
            if k in action_sig.parameters:
                call[k] = self.params[k]

        return self._Sjax(**call)


    @staticmethod
    @abstractmethod
    def _Sjax():
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
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        for fname in self.fields.keys():
            new.fields[fname] = self.fields[fname].copy()
        
        return new

    def copy(self):
        return self.__copy__()
    
    def copy_fields(self):
        new_fields = {}
        for fname in self.fields.keys():
            new_fields[fname] = self.fields[fname].copy()

        return new_fields

    def add_subaction(self, other):
        # Combine the fields; in case of name collision, make sure they are really the same field!
        for field_name in other.fields.keys():
            if self.fields.get(field_name) is not None:
                assert self.fields[field_name] is other.fields[field_name]
            
            self.fields[field_name] = other.fields[field_name]
        
        # Combine the parameters
        self.params.update(other.params)

        # If there are subactions within the other field, promote them up to the current subaction list
        if len(other.sub_actions) > 0:
            for subact in other.sub_actions:
                self.sub_actions.append(subact)
            
            other.sub_actions = []

        # Register the subaction
        self.sub_actions.append(other)

class MDIntegrator():

    def __init__(self, eps, Nstep):
        self.eps = eps
        self.Nstep = Nstep
        
        self.traj_length = eps * Nstep

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

    @partial(jax.jit, static_argnums=(0,1,2))
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

    def __init__(self, eps, Nstep, xi=0.1931833):
        self.xi = xi

        super().__init__(eps=eps, Nstep=Nstep)

    @partial(jax.jit, static_argnums=(0,1,2))
    def integrate(self, delta_X, delta_P, X, P):
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




class HMCEvolver(Evolver):

    def __init__(self, action, seed, integrator, traj_init=0, observables=None):
        self.integrator = integrator
        self.monitor = {
            'delta_H': [],
            'P_acc': [],
            'accept': [],
        }

        self.traj_init = traj_init      # Initial trajectory number
        self.traj_i = traj_init         # Current trajectory number
        self.traj_chain = [ traj_init ]

        super().__init__(action=action, seed=seed, observables=observables)

        # Avoid recreating delta functions unnecessarily
        self.make_deltas()

        # Initialize momentum fields
        self.pi_fields = {}
        for fname in self.field_names:
            self.pi_fields[fname] = self.action.fields[fname].copy()


    def H(self):
        return self._H(list(self.pi_fields.values()), self.action.S())
#        KE = self._KE(pi_list)
#        return KE + self.action.S()


    @staticmethod
    @jax.jit
    def _H(pi_fields, S_field):
        H_field = S_field
        for pi in pi_fields:
            H_field += 0.5 * pi**2

        return jnp.sum(H_field.field)
    
    @staticmethod
    @jax.jit
    def _H_density(pi_fields, S_field):
        H_field = S_field.copy()
        for pi in pi_fields:
            H_field += 0.5 * pi**2

        return H_field

    def mom_refresh(self, ntraj):
        # Refactor to try to speed up a bit...
        for fname in self.field_names:
            pi_shape = (ntraj,) + self.action.fields[fname].field.shape
            self.rng_key, fresh_pi = self._mom_heatbath(pi_shape, self.rng_key)
            self.pi_fields[fname].field = fresh_pi


    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _mom_heatbath(shape, key):
        key, subkey = jax.random.split(key)
        return key, jax.random.normal(subkey, shape=shape)
    
    # Produce an MD integrator-compatible function
    def delta_mom(self):
        F = self.action.get_forces()
        def delta_P(X, P):
            result = {}
            for field in self.field_names:
                result[field] = -1 * F(X)[field]

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
    
    @partial(jax.jit, static_argnums=(0,))
    def _MD_traj(self, fields, pi_fields):
        S_old =  self.action._S_fields(fields)
        H_old = self._H_density(list(pi_fields.values()), S_old)

        fields, pi_fields = self.integrator.integrate(
            delta_X = self.delta_X,
            delta_P = self.delta_P,
            X = fields,
            P = pi_fields,
        )

        S_new = self.action._S_fields(fields)
        H_new = self._H_density(list(pi_fields.values()), S_new)

        delta_H = jnp.sum(H_new.field - H_old.field)
        P_acc = jnp.exp(-delta_H)

        return fields, pi_fields, delta_H, P_acc

    def get_momentum(self, pi_fields, traj):
        return self._get_momentum(pi_fields, traj)

    @staticmethod
    @jax.jit
    def _get_momentum(pi_fields, traj):
        pi_traj = {}
        for fname in pi_fields.keys():
            pi_traj[fname] = pi_fields[fname].copy()
            pi_traj[fname].field = pi_traj[fname].field[traj]

        return pi_traj

    def evolve(self, ntraj=1, warmup=False):
        # Draw accept/reject numbers for this run
        self.rng_key, subkey = jax.random.split(self.rng_key)
        r_accept = np.array(jax.random.uniform(subkey, shape=(ntraj,)))

        # Heatbath momentum refresh
        self.mom_refresh(ntraj=ntraj)

        for traj in range(ntraj):

            # Store old field values
#            prev_fields = self.action.copy_fields()
            prev_fields = {fname: F.field for fname, F in self.action.fields.items() }

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
            self.action.fields, pi_traj, delta_H, P_acc = self.MD_traj(self.action.fields, pi_traj)

            self.monitor['delta_H'].append(delta_H)
            self.monitor['P_acc'].append(P_acc)

            accept = True
            if not warmup:  # Warmups always accept!
                if P_acc < 1:
#                    self.rng_key, subkey = jax.random.split(self.rng_key)
#                    r = jax.random.uniform(subkey)
                    if r_accept[traj] > P_acc:
                        accept = False
                        for fname in self.action.fields.keys():
                            self.action.fields[fname].field = prev_fields[fname]
#                        self.action.fields = prev_fields

            self.monitor['accept'].append(accept)

            # Record completed trajectory        
            for fname in self.field_names:
                self.field_chain[fname].append(self.action.fields[fname])

            self.traj_i += 1
            self.traj_chain.append(self.traj_i)

            # Measure observables
            if self.observables is not None:
                for obs in self.observables.keys():
                    obs_f, freq = self.observables[obs]

                    if (freq == 1) or (self.traj_i - self.traj_init) % freq == 0:
                        self.obs_chain[obs].append(obs_f(self.action.fields, self.action.params))

        





        





        

    
