import numpy as np
import jax.numpy as jnp

import jax
from jax.experimental.ode import odeint

class SIR_model:

    def __init__(self,
                    beta : jnp.float32,
                    gamma : jnp.float32,
                    N : jnp.int32 = 1000):
        
        self._b = beta
        self._g = gamma
        self._N = N

        # some default
        self.initial_state = jnp.array([997., 3., 0.])

        # some default
        self.times = jnp.linspace(0, 100, num=1000, dtype=jnp.float32)
    
    def set_params(self, params):
        self._b = params[0]
        self._g = params[1]
    
    def set_initial_state(self, initial_state):
        self.initial_state = initial_state
    
    def set_time_array(self, times):
        self.times = times
    
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

    def forward(self, params):
        self.set_params(params)
        return self.solve(self.initial_state, self.times)