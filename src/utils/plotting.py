"""Plotting utilities for toxicity prediction results."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_label_distribution(
    df: pd.DataFrame,
    label_cols: list[str],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot per-assay label distribution.

    Args:
        df: DataFrame with label columns.
        label_cols: List of label column names.
        save_path: Path to save the figure. If None, displays plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Calculate statistics
    stats = []
    for col in label_cols:
        n_total = len(df)
        n_valid = df[col].notna().sum()
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        n_missing = n_total - n_valid
        pos_rate = n_pos / n_valid if n_valid > 0 else 0

        stats.append({
            "assay": col,
            "positive": n_pos,
            "negative": n_neg,
            "missing": n_missing,
            "active_rate": pos_rate,
        })

    stats_df = pd.DataFrame(stats)

    # Plot 1: Stacked bar chart of label counts
    ax1 = axes[0]
    x = np.arange(len(label_cols))
    width = 0.6

    ax1.bar(x, stats_df["positive"], width, label="Positive (Active)", color="coral")
    ax1.bar(
        x, stats_df["negative"], width, bottom=stats_df["positive"],
        label="Negative (Inactive)", color="steelblue"
    )
    ax1.bar(
        x, stats_df["missing"], width,
        bottom=stats_df["positive"] + stats_df["negative"],
        label="Missing", color="lightgray"
    )

    ax1.set_xlabel("Assay")
    ax1.set_ylabel("Count")
    ax1.set_title("Label Distribution per Assay")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("NR-", "").replace("SR-", "") for c in label_cols],
                        rotation=45, ha="right")
    ax1.legend()

    # Plot 2: Active rate bar chart
    ax2 = axes[1]
    colors = ["coral" if r > 0.1 else "steelblue" for r in stats_df["active_rate"]]
    ax2.bar(x, stats_df["active_rate"] * 100, width, color=colors)
    ax2.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax2.set_xlabel("Assay")
    ax2.set_ylabel("Active Rate (%)")
    ax2.set_title("Active Rate per Assay")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("NR-", "").replace("SR-", "") for c in label_cols],
                        rotation=45, ha="right")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_metrics_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "pr_auc",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot metric comparison across models and assays.

    Args:
        results: Dictionary mapping model names to per-assay metrics.
        metric: Metric to plot ('pr_auc' or 'roc_auc').
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(results.keys())
    assays = [k for k in results[model_names[0]].keys() if k != "mean"]

    x = np.arange(len(assays))
    width = 0.8 / len(model_names)

    for i, model_name in enumerate(model_names):
        values = [results[model_name][assay][metric] for assay in assays]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name)

    ax.set_xlabel("Assay")
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_title(f"{metric.upper().replace('_', '-')} per Assay")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("NR-", "").replace("SR-", "") for a in assays],
                       rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task_name: str = "Task",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Plot ROC and PR curves for a single task.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        task_name: Name of the task for the title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    # Filter missing labels
    mask = ~np.isnan(y_true)
    y_true_f = y_true[mask].astype(int)
    y_prob_f = y_prob[mask]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC curve
    ax1 = axes[0]
    fpr, tpr, _ = roc_curve(y_true_f, y_prob_f)
    roc_auc = roc_auc_score(y_true_f, y_prob_f)
    ax1.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve - {task_name}")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # PR curve
    ax2 = axes[1]
    precision, recall, _ = precision_recall_curve(y_true_f, y_prob_f)
    pr_auc = average_precision_score(y_true_f, y_prob_f)
    baseline = y_true_f.sum() / len(y_true_f)
    ax2.plot(recall, precision, color="coral", lw=2, label=f"PR (AP = {pr_auc:.3f})")
    ax2.axhline(y=baseline, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"PR Curve - {task_name}")
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig
