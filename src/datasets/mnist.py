from __future__ import annotations
import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist( # download the mnist dataset from openml
    test_size: float = 1/7,
    seed: int = 42,
    normalize: bool = True,
    flatten: bool = True,
):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    #extract the data and target from the mnist dataset
    X = mnist.data.astype(float) #contains pixel values and y is the digit labels
    y = mnist.target.astype(int)
    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    #normalize the pixel values to be between 0 and 1
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    #flatten images into 1D vectors
    if flatten:
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_test, y_test

#just for testing
# X_train, y_train, X_test, y_test = load_mnist()

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# print(X_train[:10], y_train[:10])
# print(X_train.min(), X_train.max())  # 0.0 to 1.0
