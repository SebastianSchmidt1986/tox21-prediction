"""Evaluation metrics for toxicity prediction."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics for a single task.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels (for threshold-dependent metrics).
        y_prob: Predicted probabilities (for AUC metrics). If None, uses y_pred.
        threshold: Classification threshold for converting probabilities to labels.

    Returns:
        Dictionary containing PR-AUC, ROC-AUC, F1, precision, and recall.
    """
    metrics = {}

    # Filter out missing labels (NaN values)
    mask = ~np.isnan(y_true)
    y_true_filtered = y_true[mask].astype(int)

    if len(y_true_filtered) == 0:
        logger.warning("No valid labels to evaluate")
        return {"pr_auc": np.nan, "roc_auc": np.nan, "f1": np.nan}

    # Check if we have both classes
    unique_labels = np.unique(y_true_filtered)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in y_true: {unique_labels}")
        return {"pr_auc": np.nan, "roc_auc": np.nan, "f1": np.nan}

    if y_prob is not None:
        y_prob_filtered = y_prob[mask]
        y_pred_filtered = (y_prob_filtered >= threshold).astype(int)
    else:
        y_pred_filtered = y_pred[mask].astype(int)
        y_prob_filtered = y_pred_filtered.astype(float)

    # PR-AUC (Average Precision) - primary metric
    try:
        metrics["pr_auc"] = average_precision_score(y_true_filtered, y_prob_filtered)
    except Exception as e:
        logger.warning(f"Failed to compute PR-AUC: {e}")
        metrics["pr_auc"] = np.nan

    # ROC-AUC - secondary metric
    try:
        metrics["roc_auc"] = roc_auc_score(y_true_filtered, y_prob_filtered)
    except Exception as e:
        logger.warning(f"Failed to compute ROC-AUC: {e}")
        metrics["roc_auc"] = np.nan

    # Threshold-dependent metrics
    try:
        metrics["f1"] = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        metrics["precision"] = precision_score(
            y_true_filtered, y_pred_filtered, zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true_filtered, y_pred_filtered, zero_division=0
        )
    except Exception as e:
        logger.warning(f"Failed to compute threshold-dependent metrics: {e}")
        metrics["f1"] = np.nan
        metrics["precision"] = np.nan
        metrics["recall"] = np.nan

    return metrics


def compute_multitask_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task_names: Optional[list] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for all tasks in a multi-task setting.

    Args:
        y_true: Ground truth labels, shape (n_samples, n_tasks).
        y_prob: Predicted probabilities, shape (n_samples, n_tasks).
        task_names: List of task names. If None, uses task indices.
        threshold: Classification threshold.

    Returns:
        Dictionary mapping task names to their metrics, plus 'mean' for averages.
    """
    n_tasks = y_true.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(n_tasks)]

    results = {}
    all_pr_aucs = []
    all_roc_aucs = []

    for i, task_name in enumerate(task_names):
        task_metrics = compute_metrics(
            y_true[:, i],
            y_pred=None,
            y_prob=y_prob[:, i],
            threshold=threshold,
        )
        results[task_name] = task_metrics

        if not np.isnan(task_metrics["pr_auc"]):
            all_pr_aucs.append(task_metrics["pr_auc"])
        if not np.isnan(task_metrics["roc_auc"]):
            all_roc_aucs.append(task_metrics["roc_auc"])

    # Compute mean metrics across tasks
    results["mean"] = {
        "pr_auc": np.mean(all_pr_aucs) if all_pr_aucs else np.nan,
        "roc_auc": np.mean(all_roc_aucs) if all_roc_aucs else np.nan,
    }

    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Find optimal classification threshold on validation set.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimise ('f1', 'precision', 'recall').
        thresholds: Array of thresholds to try. Defaults to 0.1 to 0.9.

    Returns:
        Tuple of (optimal_threshold, best_metric_value).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    mask = ~np.isnan(y_true)
    y_true_filtered = y_true[mask].astype(int)
    y_prob_filtered = y_prob[mask]

    if len(y_true_filtered) == 0 or len(np.unique(y_true_filtered)) < 2:
        return 0.5, 0.0

    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_prob_filtered >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true_filtered, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true_filtered, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true_filtered, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
