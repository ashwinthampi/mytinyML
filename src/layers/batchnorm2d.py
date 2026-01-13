#batchnorm2d layer implementation
#implements 2d batch normalization for convolutional neural networks
#normalizes activations across batch dimension for each channel independently
#forward pass: normalizes using batch statistics (training) or running statistics (eval)
#backward pass: computes gradients for input, scale (gamma), and shift (beta) parameters
#improves training stability and enables deeper networks

import numpy as np

class BatchNorm2D:
    def __init__(self, num_channels: int, momentum: float = 0.9, epsilon: float = 1e-5, seed: int = 42):
        #number of channels (C dimension)
        self.num_channels = num_channels
        #momentum for running statistics update
        self.momentum = momentum
        #small constant for numerical stability
        self.epsilon = epsilon
        
        #learnable parameters: scale (gamma) and shift (beta) per channel
        rng = np.random.default_rng(seed=seed)
        self.gamma = np.ones((num_channels,), dtype=np.float32)  #scale parameter (initialized to 1)
        self.beta = np.zeros((num_channels,), dtype=np.float32)  #shift parameter (initialized to 0)
        
        #running statistics (updated during training, used during eval)
        self.running_mean = np.zeros((num_channels,), dtype=np.float32)
        self.running_var = np.ones((num_channels,), dtype=np.float32)
        
        #cache for backward pass
        self._X = None
        self._X_norm = None  #normalized input
        self._mean = None  #batch mean
        self._var = None  #batch variance
        self._std = None  #batch standard deviation
        
        #training mode flag
        self.training = True
        
        #gradients (set during backward pass)
        self.dgamma = None
        self.dbeta = None
    
    #forward pass: normalize activations
    def forward(self, X: np.ndarray) -> np.ndarray:
        #X shape: (N, C, H, W)
        #output shape: (N, C, H, W)
        
        #shape assertion
        assert X.ndim == 4, f"Input must be 4D (N, C, H, W), got {X.ndim}D"
        assert X.shape[1] == self.num_channels, f"Input channels mismatch: expected {self.num_channels}, got {X.shape[1]}"
        
        N, C, H, W = X.shape
        
        if self.training:
            #training mode: use batch statistics
            #cache input for backward pass (only in training mode)
            self._X = X
            
            #compute mean and variance across batch and spatial dimensions: (C,)
            #reshape to (N, C, H*W) for easier computation
            X_reshaped = X.reshape(N, C, -1)  #(N, C, H*W)
            
            #mean per channel: (C,)
            self._mean = np.mean(X_reshaped, axis=(0, 2))
            
            #variance per channel: (C,)
            self._var = np.mean((X_reshaped - self._mean[:, None]) ** 2, axis=(0, 2))
            
            #standard deviation: (C,)
            self._std = np.sqrt(self._var + self.epsilon)
            
            #normalize: (N, C, H, W)
            self._X_norm = (X - self._mean[None, :, None, None]) / self._std[None, :, None, None]
            
            #update running statistics (exponential moving average)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self._mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._var
        else:
            #evaluation mode: use running statistics
            #don't cache in eval mode (memory optimization, backward not called)
            self._std = np.sqrt(self.running_var + self.epsilon)
            X_norm = (X - self.running_mean[None, :, None, None]) / self._std[None, :, None, None]
            #apply scale and shift: Y = gamma * X_norm + beta
            Y = self.gamma[None, :, None, None] * X_norm + self.beta[None, :, None, None]
            return Y
        
        #apply scale and shift: Y = gamma * X_norm + beta
        Y = self.gamma[None, :, None, None] * self._X_norm + self.beta[None, :, None, None]
        
        return Y
    
    #backward pass: compute gradients
    def backward(self, dY: np.ndarray) -> np.ndarray:
        #dY shape: (N, C, H, W)
        #returns dX shape: (N, C, H, W)
        
        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._X_norm is not None, "forward() must be called before backward()"
        assert self.training, "backward() should only be called during training"
        
        #validate dY shape
        assert dY.ndim == 4, f"dY must be 4D (N, C, H, W), got {dY.ndim}D"
        assert dY.shape == self._X.shape, f"dY shape {dY.shape} doesn't match input shape {self._X.shape}"
        
        X = self._X
        X_norm = self._X_norm
        N, C, H, W = X.shape
        
        #gradient w.r.t. scale (gamma): dgamma = sum over batch and spatial dimensions of (dY * X_norm)
        self.dgamma = np.sum(dY * X_norm, axis=(0, 2, 3))
        
        #gradient w.r.t. shift (beta): dbeta = sum over batch and spatial dimensions of dY
        self.dbeta = np.sum(dY, axis=(0, 2, 3))
        
        #gradient w.r.t. normalized input: dX_norm = dY * gamma
        dX_norm = dY * self.gamma[None, :, None, None]
        
        #gradient w.r.t. input (backprop through normalization)
        #reshape for easier computation: (N, C, H*W)
        dX_norm_reshaped = dX_norm.reshape(N, C, -1)
        X_norm_reshaped = X_norm.reshape(N, C, -1)
        
        #number of elements per channel (N * H * W)
        n_elements = N * H * W
        
        #more numerically-stable backward formula (compact form)
        #dx = (1/N) * (1/std) * (N*dX_norm - sum(dX_norm) - x_norm*sum(dX_norm*x_norm))
        sum_dX_norm = np.sum(dX_norm_reshaped, axis=(0, 2), keepdims=True)  #(1, C, 1)
        sum_dX_norm_x_norm = np.sum(dX_norm_reshaped * X_norm_reshaped, axis=(0, 2), keepdims=True)  #(1, C, 1)
        
        dX_reshaped = (1.0 / n_elements) * (1.0 / self._std[None, :, None]) * (
            n_elements * dX_norm_reshaped - 
            sum_dX_norm - 
            X_norm_reshaped * sum_dX_norm_x_norm
        )
        
        #reshape back to (N, C, H, W)
        dX = dX_reshaped.reshape(N, C, H, W)
        
        return dX
    
    #return the layer parameters (scale and shift)
    def parameters(self) -> dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}
    
    #return the gradients computed during backward pass (keys match parameters for consistency)
    def gradients(self) -> dict[str, np.ndarray]:
        return {"gamma": self.dgamma, "beta": self.dbeta}
    
    #set layer to training mode (enables batch statistics and running stats update)
    def train(self) -> None:
        self.training = True
    
    #set layer to evaluation mode (uses running statistics, disables backward caching)
    def eval(self) -> None:
        self.training = False
