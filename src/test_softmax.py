#test script for softmax regression model
#loads mnist data, creates a model, and tests forward pass
#prints output shape, probability sums, and value ranges

from datasets.mnist import load_mnist
from models.softmax_regression import SoftmaxRegression
import numpy as np

X_train, y_train, X_test, y_test = load_mnist()
model = SoftmaxRegression(n_classes=10, n_features=X_train.shape[1])
probs = model.forward(X_train[:5])

print(probs.shape)
print(np.sum(probs, axis=1))
print(probs.min(), probs.max())