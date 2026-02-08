#adam optimizer implementation
#adaptive moment estimation optimizer for gradient-based optimization
#maintains per-parameter learning rates using estimates of first and second moments of gradients
#typically converges faster and requires less hyperparameter tuning than sgd for deeper networks
#optimized: in-place moment updates, precomputed bias corrections, g*g instead of g**2

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

        #precompute bias correction terms once per step (not per parameter)
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        step_size = self.lr / bc1

        for k in params:
            g = grads[k]

            #initialize moment estimates if this is the first time we see this parameter
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])

            #update biased first moment estimate (in-place to avoid allocation)
            self.m[k] *= self.beta1
            self.m[k] += (1.0 - self.beta1) * g

            #update biased second moment estimate (in-place, g*g is faster than g**2)
            self.v[k] *= self.beta2
            self.v[k] += (1.0 - self.beta2) * (g * g)

            #update parameter: param -= (lr / bc1) * m / (sqrt(v / bc2) + eps)
            denom = np.sqrt(self.v[k] * (1.0 / bc2)) + self.epsilon
            params[k] -= step_size * self.m[k] / denom
