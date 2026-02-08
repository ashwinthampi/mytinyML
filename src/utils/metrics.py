#evaluation metrics utilities
#implements confusion matrix, precision, recall, f1 score, and classification report

import numpy as np

#confusion matrix function that takes the true labels, the predicted labels, and the number of classes and returns the confusion matrix
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i, j in zip(y_true, y_pred):
        cm[i, j] += 1
    return cm

#per-class precision: TP / (TP + FP) = diagonal / column sum
def precision(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    col_sums = cm.sum(axis=0)
    prec = np.zeros(num_classes)
    for c in range(num_classes):
        if col_sums[c] > 0:
            prec[c] = cm[c, c] / col_sums[c]
    return prec

#per-class recall: TP / (TP + FN) = diagonal / row sum
def recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    row_sums = cm.sum(axis=1)
    rec = np.zeros(num_classes)
    for c in range(num_classes):
        if row_sums[c] > 0:
            rec[c] = cm[c, c] / row_sums[c]
    return rec

#per-class f1 score: 2 * precision * recall / (precision + recall)
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    prec = precision(y_true, y_pred, num_classes)
    rec = recall(y_true, y_pred, num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        if prec[c] + rec[c] > 0:
            f1[c] = 2 * prec[c] * rec[c] / (prec[c] + rec[c])
    return f1

#formatted classification report string with per-class and macro-averaged metrics
def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> str:
    prec = precision(y_true, y_pred, num_classes)
    rec = recall(y_true, y_pred, num_classes)
    f1 = f1_score(y_true, y_pred, num_classes)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    support = cm.sum(axis=1)

    lines = []
    lines.append(f"{'Class':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 55)
    for c in range(num_classes):
        lines.append(f"{c:>10} {prec[c]:>10.4f} {rec[c]:>10.4f} {f1[c]:>10.4f} {support[c]:>10}")
    lines.append("-" * 55)
    lines.append(f"{'Macro Avg':>10} {prec.mean():>10.4f} {rec.mean():>10.4f} {f1.mean():>10.4f} {support.sum():>10}")
    return "\n".join(lines)
