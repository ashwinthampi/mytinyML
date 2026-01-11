#relu activation layer implementation
#implements the rectified linear unit (relu) activation function
#forward pass: A = max(0, Z)
#backward pass: gradient is dA where Z > 0, else 0

import numpy as np

class ReLU:
    def __init__(self):
        #cache for backward pass
        self._Z = None
    
    #forward pass: compute A = max(0, Z)
    def forward(self, Z: np.ndarray) -> np.ndarray:
        self._Z = Z  #cache input for backward pass
        A = np.maximum(0.0, Z)
        return A
    
    #backward pass: compute gradient w.r.t. input
    def backward(self, dA: np.ndarray) -> np.ndarray:
        Z = self._Z
        #gradient is dA where Z > 0, else 0
        dZ = dA * (Z > 0)
        return dZ
    
    #return the layer parameters (empty for relu since it has no learnable parameters)
    def parameters(self) -> dict[str, np.ndarray]:
        return {}
