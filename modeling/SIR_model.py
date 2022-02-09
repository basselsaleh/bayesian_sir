import numpy as np
import jax.numpy as jnp

import jax
from jax.experimental.ode import odeint

class SIR_model:

    def __init__(self,
                    beta : jnp.float32 = 0.5,
                    gamma : jnp.float32 = 0.04,
                    N : jnp.int32 = 1000):
        
        self._b = beta
        self._g = gamma
        self._N = N
    
    def set_params(self, params):
        self._b = params[0]
        self._g = params[1]
    
    def rhs(self,
                state : jnp.ndarray,
                t : jnp.ndarray=None,
                ) -> jnp.ndarray:
        
        S, I, R = state

        dS_dt = - self._b * S * I / self._N
        dI_dt = self._b * S * I / self._N - self._g * I
        dR_dt = self._g * I

        return jnp.stack([dS_dt, dI_dt, dR_dt])
    
    def solve(self, initial_state, times):
        return odeint(self.rhs, initial_state, t=times, rtol=1e-10, atol=1e-10)