from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def hit_false_alarm_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """HR and FAR inspired by outbreak detection evaluation.

    HR = TP / (TP + FN)
    FAR = FP / (FP + TN)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    hr = tp / (tp + fn) if (tp + fn) else float("nan")
    far = fp / (fp + tn) if (fp + tn) else float("nan")
    return float(hr), float(far)
