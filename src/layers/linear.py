#linear layer implementation
#implements a fully-connected (linear) layer with forward and backward passes
#stores weights and biases, computes Z = X @ W + b in forward pass
#computes gradients for weights, biases, and input in backward pass

import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        #random number generator
        rng = np.random.default_rng(seed=seed)
        
        #initialize weights using he initialization
        self.W = (rng.normal(0.0, 1.0, size=(in_features, out_features)) * 
                  np.sqrt(2.0 / in_features)).astype(np.float32)
        #initialize bias vector to zeros
        self.b = np.zeros((out_features,), dtype=np.float32)
        
        #cache for backward pass
        self._X = None
        
        #gradients (set during backward pass)
        self.dW = None
        self.db = None
    
    #forward pass: compute Z = X @ W + b
    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X  #cache input for backward pass
        Z = X @ self.W + self.b
        return Z
    
    #backward pass: compute gradients for weights, biases, and input
    def backward(self, dZ: np.ndarray) -> np.ndarray:
        X = self._X
        
        #gradient w.r.t. weights and biases
        self.dW = X.T @ dZ
        self.db = np.sum(dZ, axis=0)
        
        #gradient w.r.t. input
        dX = dZ @ self.W.T
        
        return dX
    
    #return the layer parameters (weights and biases)
    def parameters(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}
    
    #return the gradients computed during backward pass (keys match parameters for consistency)
    def gradients(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}
