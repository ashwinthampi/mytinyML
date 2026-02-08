#dropout layer implementation
#implements dropout regularization to prevent overfitting
#forward pass: randomly sets activations to zero during training, scales by 1/(1-p)
#backward pass: propagates gradients only through active units
#optimized: pre-scales mask once (avoids dividing by (1-p) separately in forward and backward)
#supports GPU acceleration via CuPy backend

import numpy as np
import backend

class Dropout:
    def __init__(self, p: float = 0.5):
        #dropout probability (probability of setting a unit to zero)
        self.p = p
        #pre-compute inverse keep probability for scaling
        self._scale = 1.0 / (1.0 - p)
        #cache for backward pass (pre-scaled mask)
        self._scaled_mask = None
        #training mode flag
        self.training = True

    #forward pass: apply dropout mask during training
    def forward(self, X):
        xp = backend.xp
        if self.training:
            #generate random mask and pre-scale by 1/(1-p) once
            #reuse the same scaled mask in backward (avoids redundant division)
            mask = (xp.random.random(X.shape) > self.p).astype(np.float32)
            self._scaled_mask = mask * self._scale
            X = X * self._scaled_mask
        else:
            #during inference, no dropout applied
            self._scaled_mask = None
        return X

    #backward pass: propagate gradients only through active units
    def backward(self, dA):
        if self.training and self._scaled_mask is not None:
            #pre-scaled mask already includes the 1/(1-p) factor
            dX = dA * self._scaled_mask
        else:
            dX = dA
        return dX

    #return the layer parameters (empty for dropout since it has no learnable parameters)
    def parameters(self) -> dict:
        return {}
