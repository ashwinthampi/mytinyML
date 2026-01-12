#adam optimizer implementation
#adaptive moment estimation optimizer for gradient-based optimization
#maintains per-parameter learning rates using estimates of first and second moments of gradients
#typically converges faster and requires less hyperparameter tuning than sgd for deeper networks

import numpy as np

class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        #learning rate (step size)
        self.lr = lr
        #exponential decay rates for moment estimates
        self.beta1 = beta1
        self.beta2 = beta2
        #small constant for numerical stability
        self.epsilon = epsilon
        
        #first moment estimates (momentum-like)
        self.m = {}
        #second moment estimates (variance-like)
        self.v = {}
        #timestep counter
        self.t = 0
    
    #step function that takes the parameters and the gradients and updates the parameters
    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        #increment timestep
        self.t += 1
        
        #update each parameter
        for k in params:
            #initialize moment estimates if this is the first time we see this parameter
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            
            #update biased first moment estimate
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            
            #update biased second moment estimate
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            
            #compute bias-corrected first moment estimate
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            
            #compute bias-corrected second moment estimate
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            
            #update parameter
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
