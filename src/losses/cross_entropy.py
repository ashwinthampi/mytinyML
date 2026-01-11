#cross entropy loss function implementation
#computes cross entropy loss between predicted probabilities and true labels
#forward pass: computes the loss
#backward pass: returns gradient w.r.t. probabilities

import numpy as np
#cross entropy loss function
class CrossEntropyLoss:
    #epsilon is added to avoid log(0)
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon

    #forward pass of the loss function that takes the probabilities and the true labels and returns the loss
    def forward(self, probs: np.ndarray, y: np.ndarray) -> float:
        N = probs.shape[0]
        probs = np.clip(probs, self.epsilon, 1 - self.epsilon)
        correct_logprobs = -np.log(probs[np.arange(N), y])
        return np.mean(correct_logprobs)

    #backward pass of the loss function that takes the probabilities and the true labels and 
    #returns the gradient of the loss with respect to the probabilities
    def backward(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = probs.shape[0]
        grad = probs.copy()
        grad[np.arange(N), y] -= 1
        grad /= N
        return grad