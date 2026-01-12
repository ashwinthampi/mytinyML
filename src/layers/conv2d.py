#conv2d layer implementation
#implements 2d convolution operation for convolutional neural networks using im2col + GEMM optimization
#forward pass: uses im2col to extract sliding windows, then matrix multiplication (GEMM)
#backward pass: uses col2im to reverse im2col and vectorized gradient computation
#supports stride and padding for flexible convolution operations
#much faster than loop-based convolution (10x-100x speedup)

import numpy as np

class Conv2D:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, seed: int = 42):
        #number of input channels
        self.in_channels = in_channels
        #number of output channels (number of filters)
        self.out_channels = out_channels
        #kernel size (assumed square: kernel_size x kernel_size)
        self.kernel_size = kernel_size
        #stride for convolution
        self.stride = stride
        #padding to add to input
        self.padding = padding
        
        #random number generator
        rng = np.random.default_rng(seed=seed)
        
        #initialize weights: (out_channels, in_channels, kH, kW)
        #use he initialization adapted for conv layers
        fan_in = in_channels * kernel_size * kernel_size
        self.W = (rng.normal(0.0, 1.0, size=(out_channels, in_channels, kernel_size, kernel_size)) * 
                  np.sqrt(2.0 / fan_in)).astype(np.float32)
        #initialize bias vector to zeros
        self.b = np.zeros((out_channels,), dtype=np.float32)
        
        #cache for backward pass
        self._X = None
        self._X_shape = None
        self._X_col = None  #cache im2col result for backward pass
        self._X_padded_shape = None  #cache padded input shape
        self._H_out = None
        self._W_out = None
        
        #gradients (set during backward pass)
        self.dW = None
        self.db = None
    
    #im2col: extract sliding windows into column matrix (batch-major ordering)
    #X: (N, C_in, H, W) padded input
    #returns: (N * H_out * W_out, C_in * kernel_size * kernel_size)
    #rows are ordered: batch first, then spatial position (standard im2col layout)
    def _im2col(self, X: np.ndarray, H_out: int, W_out: int) -> np.ndarray:
        N, C_in, H_in, W_in = X.shape
        k = self.kernel_size
        s = self.stride
        
        #create output column matrix
        X_col = np.zeros((N * H_out * W_out, C_in * k * k), dtype=X.dtype)
        
        #extract each sliding window and flatten it into a row (batch-major ordering)
        for n in range(N):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    #calculate input position
                    h_start = h_out * s
                    w_start = w_out * s
                    h_end = h_start + k
                    w_end = w_start + k
                    
                    #extract patch for this batch and all channels
                    X_patch = X[n, :, h_start:h_end, w_start:w_end]  #(C_in, k, k)
                    
                    #row index: batch-major ordering (matches dY_col reshape)
                    row = n * (H_out * W_out) + (h_out * W_out + w_out)
                    
                    #flatten patch and store in column matrix
                    X_col[row] = X_patch.reshape(-1)
        
        return X_col
    
    #col2im: reverse im2col operation (accumulate gradients back to input positions)
    #dX_col: (N * H_out * W_out, C_in * kernel_size * kernel_size)
    #X_shape: (N, C_in, H, W) original input shape
    #returns: (N, C_in, H, W) gradient w.r.t. input
    #uses same batch-major ordering as im2col
    def _col2im(self, dX_col: np.ndarray, X_shape: tuple, H_out: int, W_out: int) -> np.ndarray:
        N, C_in, H_in, W_in = X_shape
        k = self.kernel_size
        s = self.stride
        
        #initialize output gradient
        dX = np.zeros(X_shape, dtype=dX_col.dtype)
        
        #accumulate gradients back to input positions (batch-major ordering)
        for n in range(N):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    #calculate input position
                    h_start = h_out * s
                    w_start = w_out * s
                    h_end = h_start + k
                    w_end = w_start + k
                    
                    #row index: batch-major ordering (matches im2col)
                    row = n * (H_out * W_out) + (h_out * W_out + w_out)
                    
                    #get column for this position
                    dX_patch = dX_col[row].reshape(C_in, k, k)
                    
                    #accumulate gradient back to input position
                    dX[n, :, h_start:h_end, w_start:w_end] += dX_patch
        
        return dX
    
    #forward pass: convolve input with filters using im2col + GEMM
    def forward(self, X: np.ndarray) -> np.ndarray:
        #X shape: (N, in_channels, H, W)
        #output shape: (N, out_channels, H_out, W_out)
        #H_out = floor((H + 2*padding - kernel_size) / stride) + 1
        #W_out = floor((W + 2*padding - kernel_size) / stride) + 1
        
        #shape assertions to prevent silent bugs
        assert X.ndim == 4, f"Input must be 4D (N, C, H, W), got {X.ndim}D"
        assert X.shape[1] == self.in_channels, f"Input channels mismatch: expected {self.in_channels}, got {X.shape[1]}"
        
        #cache input and shape for backward pass
        self._X = X
        self._X_shape = X.shape
        
        N, C_in, H_in, W_in = X.shape
        
        #calculate output dimensions
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        #cache output dimensions for backward pass
        self._H_out = H_out
        self._W_out = W_out
        
        #optional: validate that dimensions divide cleanly (helps catch accidental configs)
        #frameworks don't require this, but it helps catch mistakes
        assert (H_in + 2 * self.padding - self.kernel_size) % self.stride == 0, \
            f"Height dimension doesn't divide cleanly: (H={H_in} + 2*p={2*self.padding} - k={self.kernel_size}) % s={self.stride} != 0"
        assert (W_in + 2 * self.padding - self.kernel_size) % self.stride == 0, \
            f"Width dimension doesn't divide cleanly: (W={W_in} + 2*p={2*self.padding} - k={self.kernel_size}) % s={self.stride} != 0"
        
        #apply padding to input
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        else:
            X_padded = X
        
        #cache padded input shape for backward pass
        self._X_padded_shape = X_padded.shape
        
        #im2col: extract sliding windows into column matrix
        #X_col shape: (N * H_out * W_out, C_in * kernel_size * kernel_size)
        X_col = self._im2col(X_padded, H_out, W_out)
        
        #cache X_col for backward pass
        self._X_col = X_col
        
        #ensure contiguous arrays for optimal GEMM performance
        X_col = np.ascontiguousarray(X_col, dtype=np.float32)
        
        #flatten filters: (C_out, C_in, kH, kW) -> (C_out, C_in * kH * kW)
        W_col = self.W.reshape(self.out_channels, -1)
        W_col = np.ascontiguousarray(W_col, dtype=np.float32)
        
        #matrix multiplication: Y_col = X_col @ W_col.T + b
        #Y_col shape: (N * H_out * W_out, C_out)
        Y_col = X_col @ W_col.T + self.b
        
        #reshape output: (N * H_out * W_out, C_out) -> (N, C_out, H_out, W_out)
        Y = Y_col.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        
        return Y
    
    #backward pass: compute gradients using col2im and vectorized operations
    def backward(self, dY: np.ndarray) -> np.ndarray:
        #dY shape: (N, out_channels, H_out, W_out)
        #returns dX shape: (N, in_channels, H_in, W_in)
        
        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._X_col is not None, "forward() must be called before backward()"
        assert self._H_out is not None and self._W_out is not None, "forward() must be called before backward()"
        assert self._X_padded_shape is not None, "forward() must be called before backward()"
        
        #validate dY shape matches expected output channels
        assert dY.ndim == 4, f"dY must be 4D (N, C, H, W), got {dY.ndim}D"
        assert dY.shape[1] == self.out_channels, f"dY channels mismatch: expected {self.out_channels}, got {dY.shape[1]}"
        
        N, C_in, H_in, W_in = self._X_shape
        _, C_out, H_out, W_out = dY.shape
        
        #reshape dY to column format: (N, C_out, H_out, W_out) -> (N * H_out * W_out, C_out)
        #batch-major ordering (matches im2col)
        dY_col = dY.transpose(0, 2, 3, 1).reshape(-1, C_out)
        
        #ensure contiguous arrays for optimal GEMM performance
        dY_col = np.ascontiguousarray(dY_col, dtype=np.float32)
        
        #flatten filters: (C_out, C_in, kH, kW) -> (C_out, C_in * kH * kW)
        W_col = self.W.reshape(self.out_channels, -1)
        W_col = np.ascontiguousarray(W_col, dtype=np.float32)
        
        #gradient w.r.t. weights: dW_col = dY_col.T @ X_col
        #dW_col shape: (C_out, C_in * kH * kW)
        dW_col = dY_col.T @ self._X_col
        
        #reshape dW_col back to (C_out, C_in, kH, kW)
        self.dW = dW_col.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        #gradient w.r.t. bias: db = sum over batch and spatial positions of dY_col
        self.db = np.sum(dY_col, axis=0)
        
        #gradient w.r.t. input: dX_col = dY_col @ W_col
        #dX_col shape: (N * H_out * W_out, C_in * kH * kW)
        dX_col = dY_col @ W_col
        
        #col2im: reverse im2col to get dX_padded (use cached padded shape)
        dX_padded = self._col2im(dX_col, self._X_padded_shape, H_out, W_out)
        
        #remove padding from dX
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX
    
    #return the layer parameters (weights and biases)
    def parameters(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}
    
    #return the gradients computed during backward pass (keys match parameters for consistency)
    def gradients(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}
