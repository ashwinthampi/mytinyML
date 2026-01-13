#maxpool2d layer implementation
#implements 2d max pooling operation for convolutional neural networks
#forward pass: takes maximum value in each pool_size x pool_size region (vectorized)
#backward pass: routes gradient only to the argmax location (vectorized)
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
        self._argmax = None  #stores flat index of max position for vectorized backward
    
    #forward pass: take max in each pool_size x pool_size region (vectorized)
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
        
        #validate that dimensions divide cleanly (helps catch accidental configs)
        #frameworks don't require this, but it helps catch mistakes
        assert (H_in - self.pool_size) % self.stride == 0, \
            f"Height dimension doesn't divide cleanly: (H={H_in} - pool_size={self.pool_size}) % stride={self.stride} != 0"
        assert (W_in - self.pool_size) % self.stride == 0, \
            f"Width dimension doesn't divide cleanly: (W={W_in} - pool_size={self.pool_size}) % stride={self.stride} != 0"
        
        #vectorized pooling using reshape and max over axes
        #reshape X to extract pooling windows: (N, C, H_out, pool_size, W_out, pool_size)
        #then max over pool_size axes
        
        #create output and argmax cache
        Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
        self._argmax = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        
        #extract pooling windows using advanced indexing (vectorized)
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * self.stride
                w_start = w_out * self.stride
                h_end = h_start + self.pool_size
                w_end = w_start + self.pool_size
                
                #extract patch for all batches and channels: (N, C, pool_size, pool_size)
                X_patch = X[:, :, h_start:h_end, w_start:w_end]
                
                #reshape to flatten spatial dimensions: (N, C, pool_size * pool_size)
                X_patch_flat = X_patch.reshape(N, C, -1)
                
                #find max along flattened spatial dimension: (N, C)
                max_vals = np.max(X_patch_flat, axis=2)
                
                #find argmax (index of max): (N, C)
                max_indices = np.argmax(X_patch_flat, axis=2)
                
                #store output and argmax
                Y[:, :, h_out, w_out] = max_vals
                self._argmax[:, :, h_out, w_out] = max_indices
        
        return Y
    
    #backward pass: route gradient only to the argmax location (vectorized)
    def backward(self, dY: np.ndarray) -> np.ndarray:
        #dY shape: (N, C, H_out, W_out)
        #returns dX shape: (N, C, H_in, W_in)
        
        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._argmax is not None, "forward() must be called before backward()"
        
        #validate dY shape
        assert dY.ndim == 4, f"dY must be 4D (N, C, H, W), got {dY.ndim}D"
        assert dY.shape == self._argmax.shape, f"dY shape {dY.shape} doesn't match cached output shape {self._argmax.shape}"
        
        X = self._X
        N, C, H_in, W_in = X.shape
        _, _, H_out, W_out = dY.shape
        
        #initialize gradient (zeros everywhere, will set values at argmax positions)
        dX = np.zeros_like(X)
        
        #vectorized backward: route gradients using advanced indexing (eliminates nested loops over n and c)
        for h_out in range(H_out):
            for w_out in range(W_out):
                h_start = h_out * self.stride
                w_start = w_out * self.stride
                
                #get argmax indices for this output position: (N, C)
                max_indices = self._argmax[:, :, h_out, w_out]
                
                #get gradients for this output position: (N, C)
                dy = dY[:, :, h_out, w_out]
                
                #convert flat indices to (h, w) positions within the patch: (N, C)
                max_h = max_indices // self.pool_size
                max_w = max_indices % self.pool_size
                
                #calculate input positions: (N, C)
                h_in_positions = h_start + max_h
                w_in_positions = w_start + max_w
                
                #create index arrays for vectorized scatter-add: (N, 1) and (1, C)
                n_idx = np.arange(N)[:, None]  #(N, 1)
                c_idx = np.arange(C)[None, :]  #(1, C)
                
                #broadcast to match dy shape (N, C) for advanced indexing
                n_idx_bc = np.broadcast_to(n_idx, (N, C))
                c_idx_bc = np.broadcast_to(c_idx, (N, C))
                
                #vectorized scatter-add: route gradients to argmax positions (eliminates loops over n and c)
                np.add.at(dX, (n_idx_bc, c_idx_bc, h_in_positions, w_in_positions), dy)
        
        return dX
    
    #return the layer parameters (empty for maxpool since it has no learnable parameters)
    def parameters(self) -> dict[str, np.ndarray]:
        return {}
