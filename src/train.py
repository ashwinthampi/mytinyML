#training script for cnn model
#trains the model on mnist dataset using adam optimizer
#uses validation split and early stopping for better generalization
#evaluates on test set only at the end
#supports GPU acceleration via --device gpu flag (requires CuPy)

import argparse
import csv
import numpy as np
import time
from typing import Generator
from sklearn.model_selection import train_test_split

import backend
from datasets.mnist import load_mnist
from datasets.augmentation import augment_batch
from models.cnn import CNN
from losses.cross_entropy import CrossEntropyLoss
from optim.adam import Adam
from optim.sgd import SGD
from optim.scheduler import StepLR, CosineAnnealingLR, ReduceOnPlateau, WarmupScheduler
from utils.io import save_model
from utils.metrics import confusion_matrix, classification_report

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

    #device selection
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                        help="Device to train on (cpu or gpu, gpu requires CuPy)")

    #learning rate scheduling
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "step", "cosine", "plateau"],
                        help="Learning rate scheduler")
    parser.add_argument("--step-size", type=int, default=10,
                        help="StepLR: decay every N epochs")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="StepLR: decay factor")
    parser.add_argument("--eta-min", type=float, default=1e-6,
                        help="CosineAnnealing: minimum lr")
    parser.add_argument("--scheduler-patience", type=int, default=10,
                        help="ReduceOnPlateau: patience")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                        help="ReduceOnPlateau: reduction factor")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of warmup epochs (0 = no warmup)")
    parser.add_argument("--warmup-start-lr", type=float, default=1e-6,
                        help="Starting lr for warmup phase")

    #data augmentation
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--max-shift", type=int, default=2, help="Max pixel shift for augmentation")
    parser.add_argument("--max-angle", type=float, default=15.0, help="Max rotation angle in degrees")

    #output
    parser.add_argument("--save-path", type=str, default="cnn_mnist.npz", help="Path to save model")

    return parser.parse_args()

#accuracy function that takes the probabilities and the true labels and returns the accuracy
def accuracy(probs, y) -> float:
    xp = backend.xp
    preds = xp.argmax(probs, axis=1)
    return float(xp.mean(preds == y))

#iterate over the data in mini-batches
def iterate_minibatches(X, y, batch_size: int, seed: int = 42) -> Generator:
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

    #set device (must be done before creating model so layers use correct backend)
    if args.device == "gpu":
        backend.use_gpu()
        print("Using GPU (CuPy) backend")
    else:
        print("Using CPU (NumPy) backend")

    xp = backend.xp

    #load mnist data without flattening (for cnn: shape will be (N, 1, 28, 28))
    X_train_full, y_train_full, X_test, y_test = load_mnist(flatten=False)

    #split training data into train and validation sets (stratified to keep class distribution balanced)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=args.seed, stratify=y_train_full
    )

    #move data to device (GPU transfers data once upfront, avoids per-batch transfers)
    X_train = backend.to_device(X_train)
    X_val = backend.to_device(X_val)
    X_test = backend.to_device(X_test)
    y_train = backend.to_device(y_train)
    y_val = backend.to_device(y_val)
    y_test = backend.to_device(y_test)

    #initialize the model, loss function, and optimizer
    model = CNN(seed=args.seed, dropout=args.dropout)
    #move model parameters to device
    if backend.is_gpu():
        model.to_device()

    loss_fn = CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = Adam(lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr=args.lr)

    #create learning rate scheduler
    scheduler = None
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    elif args.scheduler == "plateau":
        scheduler = ReduceOnPlateau(optimizer, mode="min", factor=args.scheduler_factor,
                                     patience=args.scheduler_patience)

    #wrap scheduler with warmup if requested
    if args.warmup_epochs > 0:
        scheduler = WarmupScheduler(optimizer, inner_scheduler=scheduler,
                                     warmup_epochs=args.warmup_epochs,
                                     warmup_start_lr=args.warmup_start_lr)

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
    training_log = []

    #train the model
    for epoch in range(epochs):
        epoch_start_time = time.time()
        #set model to training mode (enables dropout, etc.)
        model.train()

        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            #apply data augmentation if enabled (training only)
            #augmentation runs on CPU (complex indexing), then transfers result to device
            if args.augment:
                Xb_cpu = backend.to_numpy(Xb)
                Xb_cpu = augment_batch(Xb_cpu, rng=np.random.default_rng(seed=epoch * 10000 + num_batches),
                                       max_shift=args.max_shift, max_angle=args.max_angle)
                Xb = backend.to_device(Xb_cpu)

            probs = model.forward(Xb)

            loss = loss_fn.forward(probs, yb)
            epoch_loss += loss
            num_batches += 1

            dZ2 = loss_fn.backward(probs, yb)
            grads = model.backward(dZ2)

            #track gradient norm before weight decay
            grad_norm = float(xp.sqrt(sum(xp.sum(g * g) for g in grads.values())))
            epoch_grad_norm += grad_norm

            #add l2 weight decay to gradients (only for weights, not biases)
            params = model.parameters()
            for k in grads:
                #check if key is a weight (not a bias) - handles both MLP ("W0", "W1") and CNN ("conv_W0", "linear_W0") keys
                if "_W" in k or (k.startswith("W") and len(k) > 1 and k[1].isdigit()):
                    grads[k] += weight_decay * params[k]

            #only pass learnable parameters to optimizer (excludes bn running stats)
            learnable_params = {k: params[k] for k in grads}
            optimizer.step(
                params=learnable_params,
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

        avg_grad_norm = epoch_grad_norm / num_batches

        #accumulate training log entry
        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / num_batches, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": optimizer.lr,
            "avg_grad_norm": round(avg_grad_norm, 6),
            "epoch_time": round(epoch_time, 2),
        })

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {epoch_loss/num_batches:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Grad Norm: {avg_grad_norm:.4f}, "
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

        #step the learning rate scheduler
        if scheduler is not None:
            scheduler.step(metric=val_loss)

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
    print(f"Device: {args.device.upper()}")
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

    #move predictions back to CPU for metrics
    preds = backend.to_numpy(xp.argmax(test_probs, axis=1))
    y_test_cpu = backend.to_numpy(y_test)

    cm = confusion_matrix(y_test_cpu, preds, 10)
    print("\nConfusion Matrix:")
    print(cm)

    #print classification report with per-class precision, recall, f1
    print("\nClassification Report:")
    print(classification_report(y_test_cpu, preds, 10))

    #save the model with metadata (move params to CPU for saving)
    save_params = {k: backend.to_numpy(v) for k, v in model.parameters().items()}
    metadata = {
        "model_type": "CNN",
        "architecture": {
            "conv_blocks": [
                {"in_channels": 1, "out_channels": 16, "kernel_size": 3, "padding": 1},
                {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1},
            ],
            "fc_layers": [{"in": 1568, "out": 128}, {"in": 128, "out": 10}],
            "dropout": args.dropout,
        },
        "training": {
            "epochs_trained": len(epoch_times),
            "best_val_loss": float(best_val_loss),
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "device": args.device,
        },
    }
    save_model(args.save_path, save_params, metadata=metadata)

    #save training log to csv
    if training_log:
        csv_path = args.save_path.replace(".npz", "_training_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
            writer.writeheader()
            writer.writerows(training_log)
        print(f"\nTraining log saved to {csv_path}")

if __name__ == "__main__":
    main()
