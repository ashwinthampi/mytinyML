import numpy as np

class SoftmaxRegression:
    def __init__(self, n_classes: int, n_features: int, seed: int = 42):
        #random number generator
        rng = np.random.default_rng(seed = seed)

        #weight matrix witht he shape n_classes and n_features
        self.W = rng.normal(loc = 0.0, scale = 0.01, size = (n_classes, n_features)).astype(np.float32)
        #the bias vector full of zeros 
        self.b = np.zeros((n_classes,), dtype=np.float32)
    
    #compute raw scores for each class
    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W.T + self.b

    @staticmethod
    #the softmax function that takes the raw scores and returns the probabilities
    def softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    #forward pass of the model that takes the data of the shape (n_samples, n_features) and returns the probabilities of the shape (n_samples, n_classes)
    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.softmax(self.logits(X))