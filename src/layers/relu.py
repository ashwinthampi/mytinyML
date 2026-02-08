#relu activation layer implementation
#implements the rectified linear unit (relu) activation function
#forward pass: A = max(0, Z)
#backward pass: gradient is dA where Z > 0, else 0
#optimized: caches boolean mask instead of full float32 input (8x less memory)
#supports GPU acceleration via CuPy backend

import numpy as np
import backend

class ReLU:
    def __init__(self):
        #cache boolean mask for backward pass (instead of full input tensor)
        self._mask = None

    #forward pass: compute A = max(0, Z)
    def forward(self, Z):
        #cache only the boolean mask (1 byte per element vs 4 bytes for float32)
        self._mask = Z > 0
        A = Z * self._mask
        return A

    #backward pass: compute gradient w.r.t. input
    def backward(self, dA):
        #gradient is dA where input was positive, else 0
        dZ = dA * self._mask
        return dZ

    #return the layer parameters (empty for relu since it has no learnable parameters)
    def parameters(self) -> dict:
        return {}
