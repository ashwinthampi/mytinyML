#multi-layer perceptron (mlp) implementation using composable layers
#implements a sequential-style mlp with variable depth
#uses linear and relu layers for modularity and reusability
#forward pass: loops through layers sequentially
#backward pass: propagates gradients through all layers in reverse

import numpy as np
from layers.linear import Linear
from layers.relu import ReLU

class MLP:
    def __init__(self, layer_sizes: list[int] | None = None, n_features: int | None = None, 
                 n_hidden: int | None = None, n_classes: int = 10, seed: int = 42):
        #support both new interface (layer_sizes) and old interface (n_features, n_hidden, n_classes)
        if layer_sizes is not None:
            #new interface: specify full layer architecture
            if len(layer_sizes) < 2:
                raise ValueError("layer_sizes must have at least 2 elements (input and output)")
            self.layer_sizes = layer_sizes
        else:
            #old interface: backward compatibility for 2-layer mlp
            if n_features is None or n_hidden is None:
                raise ValueError("Either layer_sizes or both n_features and n_hidden must be provided")
            self.layer_sizes = [n_features, n_hidden, n_classes]
        
        #build layers: alternate between linear and relu, with linear at the end
        self.layers = []
        linear_idx = 0
        for i in range(len(self.layer_sizes) - 1):
            #add linear layer
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            self.layers.append(Linear(in_size, out_size, seed=seed + linear_idx))
            linear_idx += 1
            
            #add relu after each linear layer except the last one
            if i < len(self.layer_sizes) - 2:
                self.layers.append(ReLU())
    
    #softmax activation function that takes raw scores and returns probabilities
    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    #forward pass through the network
    def forward(self, X: np.ndarray) -> np.ndarray:
        #loop through all layers
        x = X
        for layer in self.layers:
            x = layer.forward(x)
        
        #apply softmax to final output
        P = self.softmax(x)
        return P
    
    #backward pass through the network
    def backward(self, dZ: np.ndarray) -> dict[str, np.ndarray]:
        #backward through all layers in reverse order
        grad = dZ
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        #collect gradients from all linear layers with unique keys
        grads = {}
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer_grads = layer.gradients()
                grads[f"W{linear_idx}"] = layer_grads["W"]
                grads[f"b{linear_idx}"] = layer_grads["b"]
                linear_idx += 1
        
        return grads
    
    #return all model parameters with unique keys
    def parameters(self) -> dict[str, np.ndarray]:
        params = {}
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer_params = layer.parameters()
                params[f"W{linear_idx}"] = layer_params["W"]
                params[f"b{linear_idx}"] = layer_params["b"]
                linear_idx += 1
        return params
    
    #set model parameters from a dictionary
    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        #handle both old format (W1, b1, W2, b2) and new format (W0, b0, W1, b1, ...)
        #detect format: if W0 exists -> new format, else if W1 exists -> old format
        if "W0" not in params and "W1" in params:
            #old format: W1, b1, W2, b2 -> map to indices 0, 1
            linear_idx = 0
            for layer in self.layers:
                if isinstance(layer, Linear):
                    old_key = f"W{linear_idx + 1}"
                    old_bias_key = f"b{linear_idx + 1}"
                    if old_key in params:
                        layer.W = params[old_key]
                        layer.b = params[old_bias_key]
                    else:
                        #fallback to new format if old key not found
                        layer.W = params[f"W{linear_idx}"]
                        layer.b = params[f"b{linear_idx}"]
                    linear_idx += 1
        else:
            #new format: W0, b0, W1, b1, ...
            linear_idx = 0
            for layer in self.layers:
                if isinstance(layer, Linear):
                    layer.W = params[f"W{linear_idx}"]
                    layer.b = params[f"b{linear_idx}"]
                    linear_idx += 1
