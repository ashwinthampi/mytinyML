import numpy as np

#confusion matrix function that takes the true labels, the predicted labels, and the number of classes and returns the confusion matrix
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, j in zip(y_true, y_pred):
        cm[i, j] += 1
    return cm