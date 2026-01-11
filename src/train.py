#training script for mlp model
#trains the model on mnist dataset using stochastic gradient descent
#evaluates on train and test sets, prints metrics, and saves the trained model

import numpy as np
from typing import Generator

from datasets.mnist import load_mnist
#from models.softmax_regression import SoftmaxRegression
from models.mlp import MLP
from losses.cross_entropy import CrossEntropyLoss
from optim.sgd import SGD
from utils.io import save_model
from utils.metrics import confusion_matrix

#accuracy function that takes the probabilities and the true labels and returns the accuracy
def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y))

#iterate over the data in mini-batches
def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int = 42) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    rng = np.random.default_rng(seed=seed)
    N = X.shape[0]
    indices = np.arange(N)
    rng.shuffle(indices)

    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]

#main train function
def main():
    X_train, y_train, X_test, y_test = load_mnist()

    #initialize the model, loss function, and optimizer
    #model = SoftmaxRegression(n_classes=10, n_features=X_train.shape[1])
    model = MLP(n_features=X_train.shape[1], n_hidden=128, n_classes=10)
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(lr=0.1)

    #number of epochs and batch size
    epochs = 10
    batch_size = 128

    #train the model
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            probs = model.forward(Xb)

            loss = loss_fn.forward(probs, yb)
            epoch_loss += loss
            num_batches += 1

            # dZ = loss_fn.backward(probs, yb) 

            # dW = dZ.T @ Xb                    
            # db = np.sum(dZ, axis=0)           

            # optimizer.step(
            #     params={"W": model.W, "b": model.b},
            #     grads={"W": dW, "b": db}
            # )
            dZ2 = loss_fn.backward(probs, yb)
            grads = model.backward(dZ2)
            optimizer.step(
                params=model.parameters(),
                grads=grads
            )

        train_probs = model.forward(X_train[:5000])
        test_probs = model.forward(X_test)

        train_acc = accuracy(train_probs, y_train[:5000])
        test_acc = accuracy(test_probs, y_test)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {epoch_loss/num_batches:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Test Acc: {test_acc:.4f}"
        )

    preds = np.argmax(test_probs, axis=1)
    cm = confusion_matrix(y_test, preds, 10)
    print(cm)
    #save the model
    save_model("mlp_mnist.npz", model.parameters())

if __name__ == "__main__":
    main()
