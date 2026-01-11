import numpy as np

class SGD: #Stochastic Gradient Descent
    #learning rate is the step size for the gradient descent
    def __init__(self, lr: float = 0.1):
        self.lr = lr
    
    #step function that takes the parameters and the gradients and updates the parameters
    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for k in params:
            params[k] -= self.lr * grads[k]

    