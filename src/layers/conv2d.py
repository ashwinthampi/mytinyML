#conv2d layer implementation
#implements 2d convolution operation for convolutional neural networks
#forward pass: convolves input with learnable filters
#backward pass: computes gradients for weights, biases, and input
#supports stride and padding for flexible convolution operations

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
        self._X_padded = None
        self._X_shape = None
        
        #gradients (set during backward pass)
        self.dW = None
        self.db = None
    
    #forward pass: convolve input with filters
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
        
        #optional: validate that dimensions divide cleanly (helps catch accidental configs)
        #frameworks don't require this, but it helps catch mistakes
        assert (H_in + 2 * self.padding - self.kernel_size) % self.stride == 0, \
            f"Height dimension doesn't divide cleanly: (H={H_in} + 2*p={2*self.padding} - k={self.kernel_size}) % s={self.stride} != 0"
        assert (W_in + 2 * self.padding - self.kernel_size) % self.stride == 0, \
            f"Width dimension doesn't divide cleanly: (W={W_in} + 2*p={2*self.padding} - k={self.kernel_size}) % s={self.stride} != 0"
        
        #apply padding to input and cache for backward pass
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        else:
            X_padded = X
        #cache padded input for backward pass (avoids recomputation and potential mistakes)
        self._X_padded = X_padded
        
        #initialize output
        Y = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)
        
        #convolution using explicit loops (correctness first, optimize later)
        for n in range(N):
            for c_out in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        #calculate input position
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        #extract input patch
                        X_patch = X_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        #convolve: sum over input channels and spatial dimensions
                        conv_sum = np.sum(X_patch * self.W[c_out])
                        
                        #add bias
                        Y[n, c_out, h_out, w_out] = conv_sum + self.b[c_out]
        
        return Y
    
    #backward pass: compute gradients for weights, biases, and input
    def backward(self, dY: np.ndarray) -> np.ndarray:
        #dY shape: (N, out_channels, H_out, W_out)
        #returns dX shape: (N, in_channels, H_in, W_in)
        
        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._X_padded is not None, "forward() must be called before backward()"
        
        #validate dY shape matches expected output channels
        assert dY.ndim == 4, f"dY must be 4D (N, C, H, W), got {dY.ndim}D"
        assert dY.shape[1] == self.out_channels, f"dY channels mismatch: expected {self.out_channels}, got {dY.shape[1]}"
        
        #use cached padded input (avoids recomputation)
        X_padded = self._X_padded
        N, C_in, H_in, W_in = self._X_shape
        _, C_out, H_out, W_out = dY.shape
        
        #initialize gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(X_padded)
        
        #backward pass using explicit loops
        for n in range(N):
            for c_out in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        #calculate input position
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        #gradient w.r.t. output
                        dy = dY[n, c_out, h_out, w_out]
                        
                        #extract input patch
                        X_patch = X_padded[n, :, h_start:h_end, w_start:w_end]
                        
                        #gradient w.r.t. weights: dW = sum over batch and spatial positions of (X_patch * dY)
                        self.dW[c_out] += X_patch * dy
                        
                        #gradient w.r.t. bias: db = sum over batch and spatial positions of dY
                        self.db[c_out] += dy
                        
                        #gradient w.r.t. input: scatter gradient back to the input region using filter weights
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += self.W[c_out] * dy
        
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
