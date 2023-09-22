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

        return X, P

class HMCEvolver(Evolver):

    def __init__(self, fields, action, params, seed, observables=None):
        self.integrator = params['integrator']
        self.monitor = {
            'delta_H': [],
            'P_acc': [],
        }
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
        pi_update = []
        for i in range(len(phi)):
            pi_update.append(phi[i] + eps * pi[i])

        return pi_update

    @staticmethod
    @jax.jit
    def field_update(phi, pi, eps, force):
        delta_pi = force(phi)
        phi_update = []
        for i in range(len(phi)):
            phi_update.append(pi[i] - eps * delta_pi[i])

        return phi_update

    def register_action(self, action):
        self.action = action
        self.force = jax.grad(self.action)

    def copy_fields(self):
        copies = []
        for i in range(self.N_fields):
            copies.append(jnp.copy(self.fields[i]))
        return copies


    def evolve(self):
        # Heatbath momentum refresh
        self.mom_refresh()

        # Store old field values
        prev_fields = self.copy_fields()
        H_old = self.H()

        # Integrate the trajectory
        self.fields, self.pi_fields = self.integrator.integrate(
            x_update = self.mom_update,
            p_update = lambda X, P, eps: self.field_update(X, P, eps, self.forces),
            X = self.fields,
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
                self.fields = prev_fields

        # Measure observables (TODO)




        

    
