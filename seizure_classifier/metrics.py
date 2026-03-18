"""
Binary classification metrics used to evaluate seizure detection.

This module reimplements AUROC, AUPRC, F1, etc. in NumPy so that the
repository does not depend on scikit-learn.
"""

from __future__ import annotations

import numpy as np


def _binary_confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    y_true = (y_true > 0.5).astype(np.int32)
    y_pred = (y_prob >= thr).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC from scratch (no sklearn).
    Returns NaN if only one class present.
    """
    y_true = (y_true > 0.5).astype(np.int32)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score)  # descending
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    tpr = tps / pos
    fpr = fps / neg

    return float(np.trapz(tpr, fpr))


def average_precision_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUPRC (Average Precision) from scratch.
    Returns NaN if no positive examples.
    """
    y_true = (y_true > 0.5).astype(np.int32)
    pos = y_true.sum()
    if pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / pos

    ap = (prec[y == 1]).sum() / pos
    return float(ap)


def binary_metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5):
    tp, tn, fp, fn = _binary_confusion(y_true, y_prob, thr)

    eps = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    bal_acc = 0.5 * (rec + spec)

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "balanced_acc": bal_acc,
    }


def binary_metrics_for_assessment(
    logits: np.ndarray,
    y_true: np.ndarray,
    thr_grid=None,
    default_thr: float = 0.5,
):
    """
    Compute AUROC, AUPRC, default-threshold metrics and best-F1 threshold.
    """
    y_prob = 1.0 / (1.0 + np.exp(-logits))

    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 99)

    auroc = roc_auc_score_np(y_true, y_prob)
    auprc = average_precision_np(y_true, y_prob)

    m_def = binary_metrics_from_probs(y_true, y_prob, default_thr)

    best_f1 = -1.0
    best_thr = default_thr
    best_prec = best_rec = best_acc = None

    for thr in thr_grid:
        m = binary_metrics_from_probs(y_true, y_prob, thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
            best_prec = m["precision"]
            best_rec = m["recall"]
            best_acc = m["acc"]

    return {
        "auroc": auroc,
        "auprc": auprc,
        "acc": m_def["acc"],
        "f1": m_def["f1"],
        "precision": m_def["precision"],
        "recall": m_def["recall"],
        "best_thr": best_thr,
        "best_acc": best_acc,
        "best_f1": best_f1,
        "best_precision": best_prec,
        "best_recall": best_rec,
    }

