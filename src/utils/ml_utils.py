"""Funciones auxiliares como métricas y plotting"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def stratified_train_valid_test_split(X: pd.DataFrame, y: pd.Series, test_size: float, valid_size: float, random_state: int,) -> DatasetSplits:
    """Vamos a dividir los datos en tres (train/valid/test)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    X_train_valid = X.iloc[train_idx]
    y_train_valid = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    valid_ratio = valid_size / (1 - test_size)
    sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=random_state)
    train_idx, valid_idx = next(sss_valid.split(X_train_valid, y_train_valid))

    X_train = X_train_valid.iloc[train_idx]
    y_train = y_train_valid.iloc[train_idx]
    X_valid = X_train_valid.iloc[valid_idx]
    y_valid = y_train_valid.iloc[valid_idx]

    return DatasetSplits(X_train, X_valid, X_test, y_train, y_valid, y_test)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df = df.sort_values("score", ascending=False)
    positives = max((df["y"] == 1).sum(), 1)
    negatives = max((df["y"] == 0).sum(), 1)
    df["cum_positive"] = np.where(df["y"] == 1, 1, 0).cumsum() / positives
    df["cum_negative"] = np.where(df["y"] == 0, 1, 0).cumsum() / negatives
    return float((df["cum_positive"] - df["cum_negative"]).abs().max())


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.1) -> float:
    cutoff = int(len(y_score) * k)
    if cutoff == 0:
        return 0.0
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df = df.sort_values("score", ascending=False)
    top_k = df.head(cutoff)
    baseline_rate = df["y"].mean()
    top_rate = top_k["y"].mean()
    if baseline_rate == 0:
        return float("inf") if top_rate > 0 else 0.0
    return float(top_rate / baseline_rate)


def classification_report_from_probs(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    return report


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    metrics = classification_report_from_probs(y_true, y_score, threshold=threshold)
    metrics.update(
        {
            "roc_auc": roc_auc_score(y_true, y_score),
            "pr_auc": average_precision_score(y_true, y_score),
            "brier": brier_score_loss(y_true, y_score),
            "log_loss": log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)),
            "ks": ks_statistic(y_true, y_score),
            "lift_top_10": lift_at_k(y_true, y_score, k=0.1),
        }
    )
    return metrics


def shadow_pr_auc_drop(baseline: float, current: float) -> float:
    return baseline - current


__all__ = [
    "DatasetSplits",
    "stratified_train_valid_test_split",
    "evaluate_predictions",
    "classification_report_from_probs",
    "ks_statistic",
    "lift_at_k",
    "shadow_pr_auc_drop",
]
