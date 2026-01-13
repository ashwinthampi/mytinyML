#batchnorm1d layer implementation
#implements 1d batch normalization for fully-connected layers
#normalizes activations across batch dimension for each feature independently
#forward pass: normalizes using batch statistics (training) or running statistics (eval)
#backward pass: computes gradients for input, scale (gamma), and shift (beta) parameters
#improves training stability and enables deeper networks

import numpy as np

class BatchNorm1D:
    def __init__(self, num_features: int, momentum: float = 0.9, epsilon: float = 1e-5, seed: int = 42):
        #number of features (feature dimension)
        self.num_features = num_features
        #momentum for running statistics update
        self.momentum = momentum
        #small constant for numerical stability
        self.epsilon = epsilon
        
        #learnable parameters: scale (gamma) and shift (beta) per feature
        rng = np.random.default_rng(seed=seed)
        self.gamma = np.ones((num_features,), dtype=np.float32)  #scale parameter (initialized to 1)
        self.beta = np.zeros((num_features,), dtype=np.float32)  #shift parameter (initialized to 0)
        
        #running statistics (updated during training, used during eval)
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)
        
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
        #X shape: (N, num_features)
        #output shape: (N, num_features)
        
        #shape assertion
        assert X.ndim == 2, f"Input must be 2D (N, num_features), got {X.ndim}D"
        assert X.shape[1] == self.num_features, f"Input features mismatch: expected {self.num_features}, got {X.shape[1]}"
        
        N = X.shape[0]
        
        if self.training:
            #cache input for backward pass (only in training mode)
            self._X = X
            #training mode: use batch statistics
            #compute mean and variance across batch dimension: (num_features,)
            self._mean = np.mean(X, axis=0)
            
            #variance per feature: (num_features,)
            self._var = np.mean((X - self._mean) ** 2, axis=0)
            
            #standard deviation: (num_features,)
            self._std = np.sqrt(self._var + self.epsilon)
            
            #normalize: (N, num_features)
            self._X_norm = (X - self._mean) / self._std
            
            #update running statistics (exponential moving average)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self._mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self._var
        else:
            #evaluation mode: use running statistics
            #don't cache in eval mode (memory optimization, backward not called)
            self._std = np.sqrt(self.running_var + self.epsilon)
            X_norm = (X - self.running_mean) / self._std
            #apply scale and shift: Y = gamma * X_norm + beta
            Y = self.gamma * X_norm + self.beta
            return Y
        
        #apply scale and shift: Y = gamma * X_norm + beta
        Y = self.gamma * self._X_norm + self.beta
        
        return Y
    
    #backward pass: compute gradients
    def backward(self, dY: np.ndarray) -> np.ndarray:
        #dY shape: (N, num_features)
        #returns dX shape: (N, num_features)
        
        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._X_norm is not None, "forward() must be called before backward()"
        assert self.training, "backward() should only be called during training"
        
        #validate dY shape
        assert dY.ndim == 2, f"dY must be 2D (N, num_features), got {dY.ndim}D"
        assert dY.shape == self._X.shape, f"dY shape {dY.shape} doesn't match input shape {self._X.shape}"
        
        X = self._X
        X_norm = self._X_norm
        N = X.shape[0]
        
        #gradient w.r.t. scale (gamma): dgamma = sum over batch dimension of (dY * X_norm)
        self.dgamma = np.sum(dY * X_norm, axis=0)
        
        #gradient w.r.t. shift (beta): dbeta = sum over batch dimension of dY
        self.dbeta = np.sum(dY, axis=0)
        
        #gradient w.r.t. normalized input: dX_norm = dY * gamma
        dX_norm = dY * self.gamma
        
        #more numerically-stable backward formula (compact form)
        #dx = (1/N) * (1/std) * (N*dX_norm - sum(dX_norm) - x_norm*sum(dX_norm*x_norm))
        sum1 = np.sum(dX_norm, axis=0)  #(num_features,)
        sum2 = np.sum(dX_norm * self._X_norm, axis=0)  #(num_features,)
        
        dX = (1.0 / N) * (1.0 / self._std) * (N * dX_norm - sum1 - self._X_norm * sum2)
        
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
