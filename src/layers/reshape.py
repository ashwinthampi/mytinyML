#reshape layer implementation
#reshapes tensors to a target shape
#forward pass: reshapes input to target shape
#backward pass: reshapes gradient back to original input shape

import numpy as np

class Reshape:
    def __init__(self, target_shape: tuple[int, ...]):
        #target shape to reshape to (excluding batch dimension)
        #e.g., for (N, 784) → (N, 1, 28, 28), target_shape should be (1, 28, 28)
        self.target_shape = target_shape
        #cache for backward pass (stores original input shape)
        self._input_shape = None
    
    #forward pass: reshape input to target shape
    def forward(self, X: np.ndarray) -> np.ndarray:
        #cache original shape for backward pass
        self._input_shape = X.shape
        #reshape: combine batch dimension with target shape
        batch_size = X.shape[0]
        X_reshaped = X.reshape(batch_size, *self.target_shape)
        return X_reshaped
    
    #backward pass: reshape gradient back to original input shape
    def backward(self, dX_reshaped: np.ndarray) -> np.ndarray:
        #reshape gradient back to original input shape
        dX = dX_reshaped.reshape(self._input_shape)
        return dX
    
    #return the layer parameters (empty for reshape since it has no learnable parameters)
    def parameters(self) -> dict[str, np.ndarray]:
        return {}
        