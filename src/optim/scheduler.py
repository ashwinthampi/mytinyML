#learning rate scheduler implementations
#adjusts the optimizer learning rate during training for better convergence
#all schedulers hold a reference to the optimizer and mutate optimizer.lr via step()

import numpy as np

class StepLR:
    """Decay learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self, metric: float | None = None) -> None:
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma

    def get_lr(self) -> float:
        return self.optimizer.lr


class CosineAnnealingLR:
    """Cosine annealing schedule: lr decays from base_lr to eta_min following a cosine curve."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self, metric: float | None = None) -> None:
        self.last_epoch += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.last_epoch / self.T_max)
        ) / 2

    def get_lr(self) -> float:
        return self.optimizer.lr


class ReduceOnPlateau:
    """Reduce learning rate when a metric has stopped improving.

    Monitors a metric (e.g., validation loss) and reduces lr by factor
    when no improvement for 'patience' epochs.
    """

    def __init__(self, optimizer, mode: str = "min", factor: float = 0.1,
                 patience: int = 10, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('inf') if mode == "min" else float('-inf')
        self.num_bad_epochs = 0

    def step(self, metric: float | None = None) -> None:
        if metric is None:
            return

        improved = (self.mode == "min" and metric < self.best) or \
                   (self.mode == "max" and metric > self.best)

        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    self.optimizer.lr = new_lr
                    print(f"  -> ReduceOnPlateau: reducing lr to {new_lr:.6f}")
                self.num_bad_epochs = 0

    def get_lr(self) -> float:
        return self.optimizer.lr


class WarmupScheduler:
    """Wraps any scheduler with a linear warmup phase.

    During warmup_epochs, lr linearly increases from warmup_start_lr to base_lr.
    After warmup, delegates to the inner scheduler (if any).
    """

    def __init__(self, optimizer, inner_scheduler=None, warmup_epochs: int = 5,
                 warmup_start_lr: float = 1e-6):
        self.optimizer = optimizer
        self.inner_scheduler = inner_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self, metric: float | None = None) -> None:
        self.last_epoch += 1

        if self.last_epoch <= self.warmup_epochs:
            #linear warmup from warmup_start_lr to base_lr
            alpha = self.last_epoch / self.warmup_epochs
            self.optimizer.lr = self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr)
        else:
            #after warmup, delegate to inner scheduler
            if self.inner_scheduler is not None:
                self.inner_scheduler.step(metric=metric)

    def get_lr(self) -> float:
        return self.optimizer.lr
