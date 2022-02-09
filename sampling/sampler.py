# file for samplers, used to obtain samples from posterior in Bayesian inverse problem

import numpy as np
import scipy.stats
from tqdm import tqdm
import time

# NOTE: unlike the other samplers, this class will inherently require the use of jax
import jax
import jax.numpy as jnp
import blackjax.hmc as hmc
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup

class Sampler:
    '''
    Base class for a sampler
    '''
    def __init__(self, ndim, log_pdf):
        self.ndim = ndim
        self.log_pdf = log_pdf


class MHMCMCsampler(Sampler):
    '''
        Metropolis Hastings MCMC sampler
    '''
    def __init__(self, ndim, log_pdf, proposal_variance=1):
        '''
        Constructor

        Parameters
        -------------
        ndim                :   dimensionality of the parameter space (i.e. number of parameters being sampled)
        log_pdf             :   function which returns the log of the probability density (unnormalized)
        proposal_variance   :   variance of the Gaussian proposal density 
        '''
        super().__init__(ndim, log_pdf)
        self.proposal_variance = proposal_variance

    def run_mcmc(self, p0, num_steps, nburn=100):
        '''
        Function for running the MH-MCMC algorithm

        Parameters
        -------------
        p0          :   initial point in parameter space, from which we do our random walk
        num_steps   :   number of random walk steps to take

        Returns
        -------------
        the MCMC chain as a list, with the burn-in removed
        '''
        # create gaussian proposal density
        # use identity covariance for now, and zero mean (can just shift the mean at each proposed sample)
        proposal = scipy.stats.multivariate_normal(np.zeros(self.ndim), self.proposal_variance * np.identity(self.ndim))

        chain = [p0]
        p_current = p0
        for i in tqdm(range(num_steps)):
            # propose new sample
            p_prop = p_current + proposal.rvs(size=1)
            
            # compute acceptance probability
            a = np.exp(self.log_pdf(p_prop) - self.log_pdf(p_current))
            if np.random.rand() < a:
                chain.append(p_prop)
                p_current = p_prop
        
        return chain[nburn:]


class HamiltonianMCsampler(Sampler):
    '''
        Hamiltonian Monte Carlo Sampler
    '''
    def __init__(self, ndim, log_pdf, grad_log_pdf, L=1, dt=0.1):
        '''
        Constructor

        Parameters
        -------------
        ndim                :   dimensionality of the parameter space (i.e. number of parameters being sampled)
        log_pdf             :   function which returns the log of the probability density (unnormalized)
        grad_log_pdf        :   function which evaluates the gradient of the log probability density
        L                   :   number of leapfrog steps for the Hamiltinian integrator
        dt                  :   step size for the Hamiltonian integrator
        '''
        super().__init__(ndim, log_pdf)
        self.grad_log_pdf = grad_log_pdf
        self.L = L
        self.dt = dt

    def run_mcmc(self, p0, num_steps):
        '''
        Function for running the Hamiltonian MC algorithm

        Parameters
        -------------
        p0                  :   initial point in parameter space, from which we start our chain
        num_steps           :   number of MCMC iterations to perform

        Returns
        -------------
        the MCMC chain as a list; currently no burn-in implemented, but maybe we need it?
        '''
        # try to avoid notation confusion by renaming p to x, so that p can represent momentum in this algorithm
        if type(p0) is float:
            x0 = np.array([p0])
        else:
            x0 = p0

        dVdx = lambda x : - self.grad_log_pdf(x)

        chain = [x0]

        momentum = scipy.stats.norm(0, 1)

        # the random variable representing momentum should be n_steps x ndim
        size = (num_steps,) + (self.ndim,)
        for p_init in tqdm(momentum.rvs(size=size)):
            x_new, p_new = self._leapfrog(chain[-1], p_init, dVdx, L=self.L, dt=self.dt)

            # check metropolis hastings acceptance
            current_log_prob = -self.log_pdf(chain[-1]) - np.sum(momentum.logpdf(p_init))
            proposed_log_prob = -self.log_pdf(x_new) - np.sum(momentum.logpdf(p_new))
            a = np.exp(proposed_log_prob - current_log_prob)
            if np.random.rand() < a:
                chain.append(x_new)
        
        return chain
    
    def _leapfrog(self, x, p, dVdx, L, dt):
        '''
        Simple routine for solving Hamilton's equations with leapfrog algorithm

        Parameters
        -------------
        x       :   position of current sample (particle)
        p       :   momentum of current sample (particle)
        dVdx    :   gradient of Hamiltonian potential; this is the negative grad logpdf
        L       :   leapfrog parameter
        dt      :   step size for time integration

        Returns
        -------------
        updated position and momentum, should be x(L*dt), p(L*dt)
        '''
        x, p = np.copy(x), np.copy(p)

        p -= dVdx(x) * dt / 2
        for _ in range(int(L / dt) - 1):
            x += p * dt
            p -= dVdx(x) * dt
        x += p * dt
        p -= dVdx(x) * dt / 2

        # apparently we need to flip the sign of momentum before returning
        return x, -p


class NUTSsampler(HamiltonianMCsampler):
    '''
        No-U-Turn Sampler (NUTS), a variant of Hamiltonian MC that automatically picks L and dt
        For now (as a test), this class uses the NUTS package from https://github.com/mfouesneau/NUTS.git
    '''
    def __init__(self, ndim, log_pdf, grad_log_pdf):
        '''
        Constructor
        This sampler does not take L or dt as inputs, since these are selected automatically by the algorithm
        
        Parameters
        -------------
        ndim                :   dimensionality of the parameter space (i.e. number of parameters being sampled)
        log_pdf             :   function which returns the log of the probability density (unnormalized)
        grad_log_pdf        :   function which evaluates the gradient of the log probability density
        '''
        super().__init__(ndim, log_pdf, grad_log_pdf)

    def run_mcmc(self, p0, num_steps, num_adaptive_steps):
        '''
        Function for running the NUTS algorithm

        Parameters
        -------------
        p0                  :   initial point in parameter space, from which we start our chain
        num_steps           :   number of MCMC iterations to perform
        num_adaptive_steps  :   number of adaptive steps for the NUTS algorithm

        Returns
        -------------
        the MCMC chain as a numpy array
        '''
        import nuts

        # if p0 is a float (implying ndim=1), we need to turn it into a numpy array
        if type(p0) is float:
            assert self.ndim == 1
            p0 = np.array([p0])

        # function which returns both the logpdf and its gradient
        def pdf_with_grad_func(x):
            return self.log_pdf(x), self.grad_log_pdf(x)

        # initial step size needed by nuts6; TODO: consider letting user optionally provide this
        initial_dt = 0.2

        samples, logprob, eps = nuts.nuts6(f=pdf_with_grad_func, M=num_steps, Madapt=num_adaptive_steps,
                                            theta0=p0, delta=initial_dt, progress=True)

        return list(samples)


class BlackjaxNUTSsampler(HamiltonianMCsampler):
    '''
    Uses the Blackjax implementation of NUTS
    '''
    def __init__(self, ndim, log_pdf):
        '''
        Constructor
        This sampler does not take L or dt as inputs, since these are selected automatically by the algorithm
        
        Parameters
        -------------
        ndim                :   dimensionality of the parameter space (i.e. number of parameters being sampled)
        log_pdf             :   function which returns the log of the probability density (unnormalized)
        '''
        super().__init__(ndim, log_pdf, grad_log_pdf=None)
    
    def run_mcmc(self, p0, num_steps, rng_key, warmup_steps=1000, num_chains=1):
        '''
        Function for running the NUTS algorithm using Blackjax

        Parameters
        -------------
        p0                  :   initial point in parameter space, from which we start our chain
        num_steps           :   number of MCMC iterations to perform
        warmup_steps        :   number of steps to use for stan warmup
        num_chains          :   if > 1, p0 must be ndarray of shape (num_chains, ndim), where each row is the initial point for that chain

        Returns
        -------------
        the MCMC chain as a numpy array
        '''
        assert p0.shape == (num_chains,) +  (self.ndim,)

        # # RNG key
        # rng_key = jax.random.PRNGKey(rng_key)

        kernel_generator = lambda step_size, inv_mass_matrix: hmc.kernel(self.log_pdf, step_size, inv_mass_matrix, 30)

        # just one initial state needed for stan_warmup
        pos_for_warmup = p0 if num_chains == 1 else p0[0]
        initial_state_for_warmup = hmc.new_state(pos_for_warmup, self.log_pdf)
        print(f'Warming up for {warmup_steps} steps...')
        final_state, (step_size, inv_mass_matrix), info = stan_warmup.run(rng_key, kernel_generator, initial_state_for_warmup, warmup_steps)
        print('Finished warmup')

        kernel = nuts.kernel(self.log_pdf, step_size, inv_mass_matrix)
        kernel = jax.jit(kernel)

        # p0 is an ndarray of initial positions
        initial_states = jax.vmap(hmc.new_state, in_axes=(0, None))(p0, self.log_pdf)

        def inference_loop(rng_key, kernel, initial_states, num_samples, num_chains):
            def one_step(states, rng_key):
                keys = jax.random.split(rng_key, num_chains)
                states,_ = jax.vmap(kernel)(keys, states)
                return states
            
            keys = jax.random.split(rng_key, num_samples)
            #_,states = jax.lax.scan(one_step, initial_states, keys)
            states = [initial_states]
            for key in tqdm(keys):
                states.append(one_step(states[-1], key))

            return states
        
        states = inference_loop(rng_key, kernel, initial_states, num_steps, num_chains)

        chains = np.stack([states[i].position for i in range(len(states))], axis=1)
        return chains