"""Evaluation script for trained models on the test set."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.featurise import compute_morgan_fingerprints
from src.data.load import TOX21_ASSAYS
from src.models.baseline import PerTaskClassifier
from src.models.gnn import build_model
from src.training.train import build_pyg_dataset, evaluate_gnn_epoch
from src.utils.metrics import compute_metrics, compute_multitask_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _log_and_save_results(
    model_name: str,
    results_tuned: dict,
    output_path: Path,
) -> None:
    """Log per-task results table and save to CSV.

    Args:
        model_name: Display name for the model.
        results_tuned: Dict of per-task metrics (plus 'mean' key).
        output_path: CSV path for saving.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"TEST RESULTS — {model_name.upper()}")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"{'Task':15s}  {'PR-AUC':>8s}  {'ROC-AUC':>8s}  "
        f"{'F1':>6s}  {'Prec':>6s}  {'Recall':>6s}  {'Thresh':>6s}"
    )
    logger.info("-" * 70)
    for task_name in TOX21_ASSAYS:
        m = results_tuned[task_name]
        logger.info(
            f"{task_name:15s}  {m['pr_auc']:8.4f}  {m['roc_auc']:8.4f}  "
            f"{m['f1']:6.3f}  {m.get('precision', 0):6.3f}  "
            f"{m.get('recall', 0):6.3f}  {m.get('threshold', 0.5):6.2f}"
        )
    m = results_tuned["mean"]
    logger.info("-" * 70)
    logger.info(
        f"{'MEAN':15s}  {m['pr_auc']:8.4f}  {m['roc_auc']:8.4f}  "
        f"{m['f1']:6.3f}"
    )

    # Save CSV
    results_dir = output_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_name, task_metrics in results_tuned.items():
        row = {"model": model_name, "task": task_name}
        row.update(task_metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Saved test metrics to {output_path}")


def _compute_tuned_results(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict,
) -> dict:
    """Compute per-task metrics with tuned thresholds, plus mean.

    Args:
        y_test: Ground truth, shape (n, 12).
        y_prob: Predicted probabilities, shape (n, 12).
        thresholds: Dict mapping task names to thresholds.

    Returns:
        Results dict with per-task metrics and 'mean'.
    """
    results = {}
    for i, task_name in enumerate(TOX21_ASSAYS):
        thresh = thresholds.get(task_name, 0.5)
        task_metrics = compute_metrics(
            y_test[:, i],
            y_pred=None,
            y_prob=y_prob[:, i],
            threshold=thresh,
        )
        task_metrics["threshold"] = thresh
        results[task_name] = task_metrics

    pr_aucs = [m["pr_auc"] for m in results.values() if not np.isnan(m["pr_auc"])]
    roc_aucs = [m["roc_auc"] for m in results.values() if not np.isnan(m["roc_auc"])]
    f1s = [m["f1"] for m in results.values() if not np.isnan(m["f1"])]
    results["mean"] = {
        "pr_auc": np.mean(pr_aucs) if pr_aucs else np.nan,
        "roc_auc": np.mean(roc_aucs) if roc_aucs else np.nan,
        "f1": np.mean(f1s) if f1s else np.nan,
    }
    return results


def evaluate_baseline(
    model_path: Path,
    thresholds_path: Path | None = None,
) -> dict:
    """Evaluate a baseline model on the test set.

    Args:
        model_path: Path to saved PerTaskClassifier pickle.
        thresholds_path: Path to CSV with per-task optimal thresholds.

    Returns:
        Dictionary of per-task and mean metrics.
    """
    model = PerTaskClassifier.load(model_path)
    model_name = model.model_type

    thresholds = {}
    if thresholds_path and thresholds_path.exists():
        thresh_df = pd.read_csv(thresholds_path)
        thresholds = dict(zip(thresh_df["task"], thresh_df["threshold"]))
        logger.info(f"Loaded per-task thresholds from {thresholds_path}")

    # Load and featurise test data
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_smiles = test_df["smiles"].tolist()
    test_labels = test_df[TOX21_ASSAYS].values.astype(np.float32)

    X_test, valid_indices = compute_morgan_fingerprints(test_smiles)
    y_test = test_labels[valid_indices]

    n_dropped = len(test_smiles) - len(valid_indices)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} test molecules that failed featurisation")
    logger.info(f"Test set: {X_test.shape[0]} compounds, {X_test.shape[1]} features")

    y_prob = model.predict_proba(X_test)

    results = _compute_tuned_results(y_test, y_prob, thresholds)

    output_path = OUTPUTS_DIR / "results" / f"{model_name}_test_metrics.csv"
    _log_and_save_results(model_name, results, output_path)

    return results


def evaluate_gnn(
    checkpoint_path: Path,
    thresholds_path: Path | None = None,
) -> dict:
    """Evaluate a GNN model on the test set.

    Args:
        checkpoint_path: Path to saved .pt checkpoint.
        thresholds_path: Path to CSV with per-task thresholds.

    Returns:
        Dictionary of per-task and mean metrics.
    """
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_name = config["model"]["name"]

    logger.info(
        f"Loaded {model_name.upper()} checkpoint from epoch {checkpoint['epoch']} "
        f"(val PR-AUC={checkpoint['val_pr_auc']:.4f})"
    )

    # Load thresholds
    thresholds = {}
    if thresholds_path and thresholds_path.exists():
        thresh_df = pd.read_csv(thresholds_path)
        thresholds = dict(zip(thresh_df["task"], thresh_df["threshold"]))
        logger.info(f"Loaded per-task thresholds from {thresholds_path}")

    # Load test data and build graphs
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_smiles = test_df["smiles"].tolist()
    test_labels = test_df[TOX21_ASSAYS].values.astype(np.float32)

    test_dataset = build_pyg_dataset(test_smiles, test_labels)

    batch_size = config.get("training", {}).get("batch_size", 64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build model and load weights
    in_dim = test_dataset[0].x.shape[1]
    model = build_model(config, in_dim=in_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict
    y_test, y_prob = evaluate_gnn_epoch(model, test_loader, device)

    logger.info(f"Test set: {len(test_dataset)} compounds")

    results = _compute_tuned_results(y_test, y_prob, thresholds)

    output_path = OUTPUTS_DIR / "results" / f"{model_name}_test_metrics.csv"
    _log_and_save_results(model_name, results, output_path)

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved model file (.pkl for baselines, .pt for GNNs)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Path to CSV with per-task thresholds (optional)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_path = Path(args.model)
    thresholds_path = Path(args.thresholds) if args.thresholds else None

    if model_path.suffix == ".pkl":
        evaluate_baseline(model_path, thresholds_path)
    elif model_path.suffix == ".pt":
        evaluate_gnn(model_path, thresholds_path)
    else:
        raise ValueError(
            f"Unknown model file format: {model_path.suffix}. "
            "Expected .pkl (baseline) or .pt (GNN)."
        )


if __name__ == "__main__":
    main()
