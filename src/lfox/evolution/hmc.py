from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

class Evolver(ABC):
    # Any (MCMC) evolver needs:
    # - A list of lattice fields to be evolved
    # - An action, which is a function of the fields
    # - A dictionary of observables to measure, with measurement frequencies
    # - A dictionary of evolution parameters: integrators/solvers, hyperparameters, etc.
    # - An RNG seed
    
    # Properties it should have:
    # - A Markov chain (before evolution, just the initial
    # field configuration lives here)
    # - A dict of lists of observable measurements (paired with configuration #s)

    # Should accommodate, to start: 
    # - HMC
    # - Heatbath/OR
    # - Cluster algorithms

    def __init__(self, fields, action, params, seed, observables=None):
        self.fields = fields
        self.register_action(action)

        self.observables = observables
        self.params = params

        self.seed = seed

        self.rng_key = jax.random.PRNGKey(self.seed)
        self.N_fields = len(self.fields)

    def register_action(self, action):
        self.action = action

class VerletIntegrator():

    def __init__(self, eps, Nstep):
        self.eps = eps
        self.Nstep = Nstep
        
        self.traj_length = eps * Nstep

    @jax.jit
    def integrate(self, x_update, p_update, X, P):
        for _ in range(self.Nstep):
            X = x_update(X, P, self.eps/2.)
            P = p_update(X, P, self.eps)
            X = x_update(X, P, self.eps/2.)

class HMCEvolver(Evolver):

    def __init__(self, fields, action, params, seed, observables=None):
        self.integrator = params['integrator']
        super().__init__(fields=fields, action=action, params=params, seed=seed, observables=observables)

    def H(self):
        pi_sum = 0.0
        for pi in self.pi_fields:
            pi_sum += jnp.sum(pi)

        return 0.5 * pi_sum + self.action(*self.fields)

    def mom_refresh(self):
        for i in range(self.N_fields):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.pi_fields[i] = jax.random.normal(subkey, shape=self.fields[i].shape)
    
    @staticmethod
    @jax.jit
    def mom_update(phi, pi, eps):
        return phi + eps * pi

    @staticmethod
    @jax.jit
    def field_update(i, phi, pi, eps):
        delta_pi = self.forces[i](*self.fields)
        return self.pi_fields[i] - eps * delta_pi

    def register_action(self, action):
        self.action = action
        self.force = jax.grad(self.action)

    def evolve(self):
        # Heatbath momentum refresh
        self.mom_refresh()

        # Store old field values
        prev_fields = []
        for i in range(self.N_fields):
            prev_fields.append(jnp.copy(self.fields[i]))

        # Integrate the trajectory
        self.integrator.integrate()




        # Measure observables




        

    
