import numpy as np

class MLP:
    #initialize the MLP with the number of features, hidden layers, and classes
    def __init__(self, n_features: int, n_hidden: int, n_classes: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed=seed)

        #initialize the weights and biases for the first layer
        self.W1 = (rng.normal(0.0, 1.0, size=(n_features, n_hidden)) * np.sqrt(2.0 / n_features)).astype(np.float32)
        self.b1 = np.zeros((n_hidden,), dtype=np.float32)

        #initialize the weights and biases for the second layer
        self.W2 = (rng.normal(0.0, 1.0, size=(n_hidden, n_classes)) * np.sqrt(2.0 / n_hidden)).astype(np.float32)
        self.b2 = np.zeros((n_classes,), dtype=np.float32)

        #caches
        self._X = None
        self._Z1 = None
        self._A1 = None

    #relu activation function
    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, Z)

    #softmax activation function
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    #forward pass
    def forward(self, X: np.ndarray) -> np.ndarray:
        #calculate the input to the first layer
        Z1 = X @ self.W1 + self.b1      
        #apply the relu activation function to the first layer
        A1 = self.relu(Z1)              
        #calculate the input to the second layer
        Z2 = A1 @ self.W2 + self.b2     
        #apply the softmax activation function to the second layer
        P = self.softmax(Z2)            

        #store the inputs and activations for backpropagation
        self._X = X
        self._Z1 = Z1
        self._A1 = A1
        return P

    #backward pass
    def backward(self, dZ2: np.ndarray) -> dict[str, np.ndarray]:
        #get the inputs and activations from the caches
        X = self._X      
        Z1 = self._Z1    
        A1 = self._A1    

        #calculate the gradient of the weights and biases for the second layer
        dW2 = A1.T @ dZ2                 
        db2 = np.sum(dZ2, axis=0)        

        #calculate the gradient of the weights and biases for the first layer
        dA1 = dZ2 @ self.W2.T            
        dZ1 = dA1 * (Z1 > 0)             

        #calculate the gradient of the weights and biases for the first layer
        dW1 = X.T @ dZ1                  
        db1 = np.sum(dZ1, axis=0)        

        #return the gradients of the weights and biases
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    #return the weights and biases
    def parameters(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
