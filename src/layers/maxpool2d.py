#maxpool2d layer implementation
#implements 2d max pooling operation for convolutional neural networks
#forward pass: takes maximum value in each pool_size x pool_size region
#backward pass: routes gradient only to the argmax location (the position that had the max value)
#reduces spatial dimensions, improves translation invariance, and speeds up computation

import numpy as np

class MaxPool2D:
    def __init__(self, pool_size: int = 2, stride: int | None = None):
        #pool size (assumed square: pool_size x pool_size)
        self.pool_size = pool_size
        #stride for pooling (defaults to pool_size for non-overlapping pooling)
        self.stride = stride if stride is not None else pool_size
        
        #cache for backward pass (stores which position had the max value)
        self._X = None
        self._argmax = None
    
    #forward pass: take max in each pool_size x pool_size region
    def forward(self, X: np.ndarray) -> np.ndarray:
        #X shape: (N, C, H, W)
        #output shape: (N, C, H_out, W_out)
        #H_out = floor((H - pool_size) / stride) + 1
        #W_out = floor((W - pool_size) / stride) + 1
        
        #shape assertion
        assert X.ndim == 4, f"Input must be 4D (N, C, H, W), got {X.ndim}D"
        
        #cache input for backward pass
        self._X = X
        
        N, C, H_in, W_in = X.shape
        
        #calculate output dimensions
        H_out = (H_in - self.pool_size) // self.stride + 1
        W_out = (W_in - self.pool_size) // self.stride + 1
        
        #optional: validate that dimensions divide cleanly (helps catch accidental configs)
        #frameworks don't require this, but it helps catch mistakes
        assert (H_in - self.pool_size) % self.stride == 0, \
            f"Height dimension doesn't divide cleanly: (H={H_in} - pool_size={self.pool_size}) % stride={self.stride} != 0"
        assert (W_in - self.pool_size) % self.stride == 0, \
            f"Width dimension doesn't divide cleanly: (W={W_in} - pool_size={self.pool_size}) % stride={self.stride} != 0"
        
        #initialize output and argmax cache
        Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
        #cache stores the position (h, w) in the pool region that had the max value
        self._argmax = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)
        
        #max pooling using explicit loops
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        #calculate input position
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        #extract input patch
                        X_patch = X[n, c, h_start:h_end, w_start:w_end]
                        
                        #find max value and its position in the patch
                        max_val = np.max(X_patch)
                        #find argmax (position of max value) in flattened patch
                        max_flat_idx = np.argmax(X_patch)
                        #convert flat index to (h, w) position within the patch
                        max_h = max_flat_idx // self.pool_size
                        max_w = max_flat_idx % self.pool_size
                        
                        #store output
                        Y[n, c, h_out, w_out] = max_val
                        #cache argmax position for backward pass
                        self._argmax[n, c, h_out, w_out] = [max_h, max_w]
        
        return Y
    
    #backward pass: route gradient only to the argmax location
    def backward(self, dY: np.ndarray) -> np.ndarray:
        #dY shape: (N, C, H_out, W_out)
        #returns dX shape: (N, C, H_in, W_in)
        
        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._argmax is not None, "forward() must be called before backward()"
        
        #validate dY shape
        assert dY.ndim == 4, f"dY must be 4D (N, C, H, W), got {dY.ndim}D"
        assert dY.shape == self._argmax.shape[:4], f"dY shape {dY.shape} doesn't match cached output shape {self._argmax.shape[:4]}"
        
        X = self._X
        N, C, H_in, W_in = X.shape
        _, _, H_out, W_out = dY.shape
        
        #initialize gradient (zeros everywhere, will set values at argmax positions)
        dX = np.zeros_like(X)
        
        #backward pass: route gradient only to argmax locations
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        #calculate input position
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        
                        #get argmax position (where the max value was in the patch)
                        max_h, max_w = self._argmax[n, c, h_out, w_out]
                        
                        #route gradient to the position that had the max value
                        dX[n, c, h_start + max_h, w_start + max_w] += dY[n, c, h_out, w_out]
        
        return dX
    
    #return the layer parameters (empty for maxpool since it has no learnable parameters)
    def parameters(self) -> dict[str, np.ndarray]:
        return {}
