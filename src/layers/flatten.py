#flatten layer implementation
#flattens multi-dimensional tensors to 2D for fully-connected layers
#forward pass: (N, C, H, W) → (N, C*H*W)
#backward pass: reshapes gradient back to original input shape

import numpy as np

class Flatten:
    def __init__(self):
        #cache for backward pass (stores original input shape)
        self._input_shape = None
    
    #forward pass: flatten all dimensions except the first (batch dimension)
    def forward(self, X: np.ndarray) -> np.ndarray:
        #cache original shape for backward pass
        self._input_shape = X.shape
        #flatten: (N, ...) → (N, -1) keeps batch dimension, flattens the rest
        batch_size = X.shape[0]
        X_flat = X.reshape(batch_size, -1)
        return X_flat
    
    #backward pass: reshape gradient back to original input shape
    def backward(self, dX_flat: np.ndarray) -> np.ndarray:
        #reshape gradient back to original input shape
        dX = dX_flat.reshape(self._input_shape)
        return dX
    
    #return the layer parameters (empty for flatten since it has no learnable parameters)
    def parameters(self) -> dict[str, np.ndarray]:
        return {}
