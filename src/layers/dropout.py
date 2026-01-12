#dropout layer implementation
#implements dropout regularization to prevent overfitting
#forward pass: randomly sets activations to zero during training, scales by 1/(1-p) during inference
#backward pass: propagates gradients only through active units

import numpy as np

class Dropout:
    def __init__(self, p: float = 0.5):
        #dropout probability (probability of setting a unit to zero)
        self.p = p
        #cache for backward pass
        self._mask = None
        #training mode flag
        self.training = True
    
    #forward pass: apply dropout mask during training
    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.training:
            #generate random mask: 1 with probability (1-p), 0 with probability p
            self._mask = (np.random.random(X.shape) > self.p).astype(np.float32)
            #scale by 1/(1-p) to maintain expected value during training
            X = X * self._mask / (1 - self.p)
        else:
            #during inference, no dropout applied
            self._mask = None
        return X
    
    #backward pass: propagate gradients only through active units
    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.training and self._mask is not None:
            #only propagate gradients through active units, scale by 1/(1-p)
            dX = dA * self._mask / (1 - self.p)
        else:
            #during inference or if mask not set, pass through unchanged
            dX = dA
        return dX
    
    #return the layer parameters (empty for dropout since it has no learnable parameters)
    def parameters(self) -> dict[str, np.ndarray]:
        return {}
