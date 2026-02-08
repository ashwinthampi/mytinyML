#convolutional neural network (cnn) implementation using composable layers
#implements a sequential-style cnn with convolutional and fully-connected layers
#uses conv2d, relu, maxpool2d, flatten, linear, and dropout layers
#forward pass: loops through layers sequentially
#backward pass: propagates gradients through all layers in reverse

import numpy as np
from layers.conv2d import Conv2D
from layers.linear import Linear
from layers.relu import ReLU
from layers.maxpool2d import MaxPool2D
from layers.flatten import Flatten
from layers.dropout import Dropout
from layers.batchnorm2d import BatchNorm2D

class CNN:
    def __init__(self, seed: int = 42, dropout: float = 0.2):
        #build layers: conv layers -> pooling -> flatten -> fully-connected layers
        self.layers = []

        #conv block 1: conv -> batchnorm -> relu -> pool
        self.layers.append(Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, seed=seed))
        self.layers.append(BatchNorm2D(num_channels=16))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))

        #conv block 2: conv -> batchnorm -> relu -> pool
        self.layers.append(Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, seed=seed + 1))
        self.layers.append(BatchNorm2D(num_channels=32))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))

        #flatten: (N, C, H, W) -> (N, C*H*W)
        self.layers.append(Flatten())

        #fully-connected layers: linear -> relu -> dropout -> linear
        self.layers.append(Linear(in_features=32 * 7 * 7, out_features=128, seed=seed + 2))
        self.layers.append(ReLU())
        self.layers.append(Dropout(p=dropout))
        self.layers.append(Linear(in_features=128, out_features=10, seed=seed + 3))
    
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
        
        #collect gradients from all parameterized layers (conv2d, batchnorm, and linear) with unique keys
        grads = {}
        conv_idx = 0
        bn_idx = 0
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer_grads = layer.gradients()
                grads[f"conv_W{conv_idx}"] = layer_grads["W"]
                grads[f"conv_b{conv_idx}"] = layer_grads["b"]
                conv_idx += 1
            elif isinstance(layer, BatchNorm2D):
                layer_grads = layer.gradients()
                grads[f"bn_gamma{bn_idx}"] = layer_grads["gamma"]
                grads[f"bn_beta{bn_idx}"] = layer_grads["beta"]
                bn_idx += 1
            elif isinstance(layer, Linear):
                layer_grads = layer.gradients()
                grads[f"linear_W{linear_idx}"] = layer_grads["W"]
                grads[f"linear_b{linear_idx}"] = layer_grads["b"]
                linear_idx += 1
        
        return grads
    
    #return all model parameters with unique keys (including batch norm running statistics)
    def parameters(self) -> dict[str, np.ndarray]:
        params = {}
        conv_idx = 0
        bn_idx = 0
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer_params = layer.parameters()
                params[f"conv_W{conv_idx}"] = layer_params["W"]
                params[f"conv_b{conv_idx}"] = layer_params["b"]
                conv_idx += 1
            elif isinstance(layer, BatchNorm2D):
                layer_params = layer.parameters()
                params[f"bn_gamma{bn_idx}"] = layer_params["gamma"]
                params[f"bn_beta{bn_idx}"] = layer_params["beta"]
                #also save running statistics for inference
                params[f"bn_running_mean{bn_idx}"] = layer.running_mean
                params[f"bn_running_var{bn_idx}"] = layer.running_var
                bn_idx += 1
            elif isinstance(layer, Linear):
                layer_params = layer.parameters()
                params[f"linear_W{linear_idx}"] = layer_params["W"]
                params[f"linear_b{linear_idx}"] = layer_params["b"]
                linear_idx += 1
        return params
    
    #set model to training mode (enables dropout, batch norm training, etc.)
    def train(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
    
    #set model to evaluation mode (disables dropout, uses batch norm running stats, etc.)
    def eval(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
    
    #set model parameters from a dictionary (including batch norm running statistics)
    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        conv_idx = 0
        bn_idx = 0
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                layer.W = params[f"conv_W{conv_idx}"]
                layer.b = params[f"conv_b{conv_idx}"]
                conv_idx += 1
            elif isinstance(layer, BatchNorm2D):
                layer.gamma = params[f"bn_gamma{bn_idx}"]
                layer.beta = params[f"bn_beta{bn_idx}"]
                #also restore running statistics for inference
                if f"bn_running_mean{bn_idx}" in params:
                    layer.running_mean = params[f"bn_running_mean{bn_idx}"]
                if f"bn_running_var{bn_idx}" in params:
                    layer.running_var = params[f"bn_running_var{bn_idx}"]
                bn_idx += 1
            elif isinstance(layer, Linear):
                layer.W = params[f"linear_W{linear_idx}"]
                layer.b = params[f"linear_b{linear_idx}"]
                linear_idx += 1
