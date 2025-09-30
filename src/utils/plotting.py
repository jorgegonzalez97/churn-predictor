"""Funciones auxiliares como métricas y plotting"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc(y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_precision_recall(y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Calibration Curve")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def plot_lift(y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cumulative = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    deciles = np.linspace(0, 1, len(cumulative))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(deciles, cumulative, label="Cumulative gains")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Baseline")
    ax.set_xlabel("Population share")
    ax.set_ylabel("Cumulative positive rate")
    ax.set_title("Cumulative Gains / Lift")
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig

# Garantizamos path, ajustamos márgenes, guardamos en disco y cerramos img para no consumir memoria
def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = ["plot_roc", "plot_precision_recall", "plot_calibration", "plot_lift", "save_figure"]
