#training script for cnn model
#trains the model on mnist dataset using adam optimizer
#uses validation split and early stopping for better generalization
#evaluates on test set only at the end

import argparse
import numpy as np
import time
from typing import Generator
from sklearn.model_selection import train_test_split

from datasets.mnist import load_mnist
from models.cnn import CNN
from losses.cross_entropy import CrossEntropyLoss
from optim.adam import Adam
from optim.sgd import SGD
from utils.io import save_model
from utils.metrics import confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on MNIST")

    #training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    #optimizer selection
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer to use")

    #model
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    #output
    parser.add_argument("--save-path", type=str, default="cnn_mnist.npz", help="Path to save model")

    return parser.parse_args()

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
    args = parse_args()

    #load mnist data without flattening (for cnn: shape will be (N, 1, 28, 28))
    X_train_full, y_train_full, X_test, y_test = load_mnist(flatten=False)

    #split training data into train and validation sets (stratified to keep class distribution balanced)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=args.seed, stratify=y_train_full
    )

    #initialize the model, loss function, and optimizer
    model = CNN(seed=args.seed, dropout=args.dropout)
    loss_fn = CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr=args.lr)

    #hyperparameters from args
    epochs = args.epochs
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    patience = args.patience

    #early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_params = None

    #track training time
    training_start_time = time.time()
    epoch_times = []

    #train the model
    for epoch in range(epochs):
        epoch_start_time = time.time()
        #set model to training mode (enables dropout, etc.)
        model.train()

        epoch_loss = 0.0
        num_batches = 0

        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            probs = model.forward(Xb)

            loss = loss_fn.forward(probs, yb)
            epoch_loss += loss
            num_batches += 1

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

        #evaluate on training and validation sets (use subset for speed)
        train_probs = model.forward(X_train[:5000])
        val_probs = model.forward(X_val[:5000])  #evaluate on subset for speed (full val is expensive)

        train_acc = accuracy(train_probs, y_train[:5000])
        val_acc = accuracy(val_probs, y_val[:5000])
        val_loss = loss_fn.forward(val_probs, y_val[:5000])

        #calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {epoch_loss/num_batches:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Time: {epoch_time:.2f}s"
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

    #calculate total training time
    total_training_time = time.time() - training_start_time

    #restore best model parameters
    if best_model_params is not None:
        model.set_parameters(best_model_params)
        print(f"\nRestored best model (val loss: {best_val_loss:.4f})")

    #print training time summary
    print("\n" + "="*60)
    print("TRAINING TIME SUMMARY")
    print("="*60)
    if epoch_times:
        avg_epoch_time = np.mean(epoch_times)
        print(f"Total epochs trained: {len(epoch_times)}")
        print(f"Average time per epoch: {avg_epoch_time:.2f}s ({avg_epoch_time/60:.2f} min)")
        print(f"Fastest epoch: {min(epoch_times):.2f}s")
        print(f"Slowest epoch: {max(epoch_times):.2f}s")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} min / {total_training_time/3600:.2f} hours)")
    print("="*60 + "\n")

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
    save_model(args.save_path, model.parameters())

if __name__ == "__main__":
    main()
