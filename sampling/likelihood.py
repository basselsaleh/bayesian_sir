# this file will be for a general use likelihood class for bayesian inference

import numpy as np


class Likelihood:
    '''
    Base class for a general likelihood
    '''
    def __init__(self):
        pass


class GaussianLikelihood:
    '''
    Gaussian likelihood corresponding to an additive gaussian noise assumption
    '''
    def __init__(self, param_dim, loss_func):
        '''
        Constructor

        Parameters
        -------------
        param_dim    :   dimensionality of the parameter space (i.e. number of parameters being sampled)
        loss_func    :   function which evaluates the loss, given a value of the parameter; this should have the data implicit within it
        '''
        self.param_dim = param_dim
        self.loss_func = loss_func

    def log_likelihood(self, param):
        '''
        This computes the log of the likelihood, which for a gaussian likelihood is just -0.5 * the loss (aka misfit)

        Parameters
        -------------
        param      :   value of the parameter at which we want to evaluate the log_likelihood

        Returns
        -------------
        -0.5 * the loss function
        '''
        # NOTE: we're absorbing a lot of the assumptions about noise etc. in how the loss is defined. in particular, if the loss
        # is just the unweighted L2 misfit between the model and the data, this corresponds to zero-mean white, stationary, gaussian noise 
        # with pointwise variance = 1 (i.e. noise covariance is identity)
        return -0.5 * self.loss_func(param)
    
    def likelihood(self, param):
        return np.exp(self.log_likelihood(param))
