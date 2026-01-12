#training script for cnn model
#trains the model on mnist dataset using adam optimizer
#uses validation split and early stopping for better generalization
#evaluates on test set only at the end

import numpy as np
from typing import Generator
from sklearn.model_selection import train_test_split

from datasets.mnist import load_mnist
#from models.softmax_regression import SoftmaxRegression
#from models.mlp import MLP
from models.cnn import CNN
from losses.cross_entropy import CrossEntropyLoss
from optim.adam import Adam
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
    #load mnist data without flattening (for cnn: shape will be (N, 1, 28, 28))
    X_train_full, y_train_full, X_test, y_test = load_mnist(flatten=False)
    
    #split training data into train and validation sets (stratified to keep class distribution balanced)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    #initialize the model, loss function, and optimizer
    #model = SoftmaxRegression(n_classes=10, n_features=X_train.shape[1])
    #use cnn model for image classification
    model = CNN()
    loss_fn = CrossEntropyLoss()
    #optimizer = SGD(lr=0.1)
    optimizer = Adam(lr=0.001)

    #number of epochs and batch size
    epochs = 50
    batch_size = 128
    
    #l2 weight decay parameter
    weight_decay = 1e-4
    
    #early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_params = None

    #train the model
    for epoch in range(epochs):
        #set model to training mode (enables dropout, etc.)
        model.train()
        
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
            
            #add l2 weight decay to gradients (only for weights, not biases)
            params = model.parameters()
            for k in grads:
                #check if key is a weight (not a bias) - handles both MLP ("W0", "W1") and CNN ("conv_W0", "linear_W0") keys
                if "_W" in k or (k.startswith("W") and len(k) > 1 and k[1].isdigit()):
                    grads[k] += weight_decay * params[k]
            
            optimizer.step(
                params=params,
                grads=grads
            )

        #set model to evaluation mode (disables dropout, etc.)
        model.eval()
        
        #evaluate on training and validation sets
        train_probs = model.forward(X_train[:5000])
        val_probs = model.forward(X_val)
        
        train_acc = accuracy(train_probs, y_train[:5000])
        val_acc = accuracy(val_probs, y_val)
        val_loss = loss_fn.forward(val_probs, y_val)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {epoch_loss/num_batches:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )
        
        #early stopping: check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            #save best model parameters
            best_model_params = {k: v.copy() for k, v in model.parameters().items()}
            print(f"  -> New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered after {epoch+1} epochs")
                break

    #restore best model parameters
    if best_model_params is not None:
        model.set_parameters(best_model_params)
        print(f"\nRestored best model (val loss: {best_val_loss:.4f})")
    
    #evaluate on test set (only at the end)
    model.eval()
    print("\nEvaluating on test set...")
    test_probs = model.forward(X_test)
    test_acc = accuracy(test_probs, y_test)
    test_loss = loss_fn.forward(test_probs, y_test)
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    preds = np.argmax(test_probs, axis=1)
    cm = confusion_matrix(y_test, preds, 10)
    print("\nConfusion Matrix:")
    print(cm)
    
    #save the model
    save_model("cnn_mnist.npz", model.parameters())

if __name__ == "__main__":
    main()
