#multi-layer perceptron (mlp) implementation using composable layers
#implements a 2-layer mlp with one hidden layer
#uses linear and relu layers for modularity and reusability
#forward pass: input -> linear1 -> relu1 -> linear2 -> softmax
#backward pass: propagates gradients through all layers

import numpy as np
from layers.linear import Linear
from layers.relu import ReLU

class MLP:
    def __init__(self, n_features: int, n_hidden: int, n_classes: int = 10, seed: int = 42):
        #first layer: input -> hidden
        self.linear1 = Linear(n_features, n_hidden, seed=seed)
        self.relu1 = ReLU()
        
        #second layer: hidden -> output
        self.linear2 = Linear(n_hidden, n_classes, seed=seed + 1)
    
    #softmax activation function that takes raw scores and returns probabilities
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    #forward pass through the network
    def forward(self, X: np.ndarray) -> np.ndarray:
        #first layer: linear -> relu
        Z1 = self.linear1.forward(X)
        A1 = self.relu1.forward(Z1)
        
        #second layer: linear -> softmax
        Z2 = self.linear2.forward(A1)
        P = self.softmax(Z2)
        
        return P
    
    #backward pass through the network
    def backward(self, dZ2: np.ndarray) -> dict[str, np.ndarray]:
        #backward through linear2
        dA1 = self.linear2.backward(dZ2)
        grads2 = self.linear2.gradients()
        
        #backward through relu1
        dZ1 = self.relu1.backward(dA1)
        
        #backward through linear1
        _ = self.linear1.backward(dZ1)
        grads1 = self.linear1.gradients()
        
        #aggregate all gradients (map from layer keys to mlp keys)
        return {
            "W1": grads1["W"],
            "b1": grads1["b"],
            "W2": grads2["W"],
            "b2": grads2["b"]
        }
    
    #return all model parameters
    def parameters(self) -> dict[str, np.ndarray]:
        params1 = self.linear1.parameters()
        params2 = self.linear2.parameters()
        
        return {
            "W1": params1["W"],
            "b1": params1["b"],
            "W2": params2["W"],
            "b2": params2["b"]
        }
    
    #set model parameters from a dictionary
    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.linear1.W = params["W1"]
        self.linear1.b = params["b1"]
        self.linear2.W = params["W2"]
        self.linear2.b = params["b2"]
