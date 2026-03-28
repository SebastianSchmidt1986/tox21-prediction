"""Training script for baseline and GNN models."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.featurise import compute_morgan_fingerprints, smiles_to_pyg_data
from src.data.load import TOX21_ASSAYS
from src.models.baseline import PerTaskClassifier
from src.models.gnn import build_model
from src.models.multitask import MaskedBCELoss, compute_pos_weights
from src.utils.metrics import compute_multitask_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_split_data(
    split_name: str,
    label_cols: list[str] = TOX21_ASSAYS,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load a data split and return the DataFrame, SMILES, and labels.

    Args:
        split_name: One of 'train', 'val', 'test'.
        label_cols: List of label column names.

    Returns:
        Tuple of (DataFrame, smiles_list, label_array).
        label_array has shape (n_samples, n_tasks) with NaN for missing.
    """
    path = SPLITS_DIR / f"{split_name}.csv"
    df = pd.read_csv(path)

    smiles_list = df["smiles"].tolist()
    labels = df[label_cols].values.astype(np.float32)

    logger.info(f"Loaded {split_name}: {len(df)} compounds")
    return df, smiles_list, labels


def featurise_fingerprints(
    smiles_list: list[str],
    labels: np.ndarray,
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Morgan fingerprints and align with labels.

    Drops any molecules that fail featurisation.

    Args:
        smiles_list: List of SMILES strings.
        labels: Label array, shape (n_samples, n_tasks).
        radius: Morgan fingerprint radius.
        n_bits: Number of fingerprint bits.

    Returns:
        Tuple of (fingerprints, labels) with failed molecules removed.
    """
    fps, valid_indices = compute_morgan_fingerprints(
        smiles_list, radius=radius, n_bits=n_bits
    )
    labels_valid = labels[valid_indices]

    n_dropped = len(smiles_list) - len(valid_indices)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} molecules that failed featurisation")

    logger.info(f"Fingerprints shape: {fps.shape}")
    return fps, labels_valid


# ---------------------------------------------------------------------------
# PyG dataset construction
# ---------------------------------------------------------------------------

def build_pyg_dataset(
    smiles_list: list[str],
    labels: np.ndarray,
) -> list:
    """Convert SMILES to a list of PyTorch Geometric Data objects.

    Args:
        smiles_list: List of SMILES strings.
        labels: Label array, shape (n_samples, n_tasks). NaN = missing.

    Returns:
        List of PyG Data objects (molecules that failed conversion are skipped).
    """
    dataset = []
    n_failed = 0
    for i, smiles in enumerate(smiles_list):
        data = smiles_to_pyg_data(smiles, y=labels[i])
        if data is not None:
            dataset.append(data)
        else:
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"Skipped {n_failed} molecules that failed graph conversion")
    logger.info(f"Built PyG dataset with {len(dataset)} graphs")

    return dataset


# ---------------------------------------------------------------------------
# GNN training
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_gnn_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader, returning labels and probabilities.

    Args:
        model: GNN model.
        loader: PyG DataLoader.
        device: Torch device.

    Returns:
        Tuple of (all_labels, all_probs), each shape (n_samples, n_tasks).
    """
    model.eval()
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.sigmoid(logits)

        all_labels.append(batch.y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.vstack(all_labels), np.vstack(all_probs)


def train_gnn(config: dict) -> None:
    """Train a GNN model (GCN or GIN).

    Args:
        config: Configuration dictionary from YAML file.
    """
    from torch_geometric.loader import DataLoader

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_cfg = config["model"]
    model_name = model_cfg["name"]
    training_cfg = config["training"]
    data_cfg = config.get("data", {})
    loss_cfg = training_cfg.get("loss", {})

    # Load splits
    logger.info("Loading data splits...")
    _, train_smiles, train_labels = load_split_data("train")
    _, val_smiles, val_labels = load_split_data("val")

    # Build PyG datasets
    logger.info("Building molecular graphs...")
    train_dataset = build_pyg_dataset(train_smiles, train_labels)
    val_dataset = build_pyg_dataset(val_smiles, val_labels)

    # DataLoaders
    batch_size = training_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False) and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Determine input feature dim from first graph
    in_dim = train_dataset[0].x.shape[1]
    logger.info(f"Node feature dimensionality: {in_dim}")

    # Build model
    model = build_model(config, in_dim=in_dim).to(device)

    # Loss function
    pos_weight = None
    if loss_cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weights(train_labels, max_weight=50.0).to(device)
        logger.info(f"Using per-task pos_weight (max={pos_weight.max():.1f})")

    criterion = MaskedBCELoss(pos_weight=pos_weight).to(device)

    # Optimiser
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 0.001),
        weight_decay=training_cfg.get("weight_decay", 0.0001),
    )

    # LR scheduler
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="max",
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 10),
        min_lr=scheduler_cfg.get("min_lr", 1e-5),
    )

    # Early stopping
    es_cfg = training_cfg.get("early_stopping", {})
    es_patience = es_cfg.get("patience", 20)
    es_metric = es_cfg.get("metric", "val_pr_auc")

    epochs = training_cfg.get("epochs", 100)
    log_every = config.get("logging", {}).get("log_every_n_steps", 50)
    save_checkpoints = config.get("logging", {}).get("save_checkpoints", True)

    models_dir = OUTPUTS_DIR / "models"
    results_dir = OUTPUTS_DIR / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_metric = -float("inf")
    best_epoch = 0
    patience_counter = 0

    logger.info("=" * 60)
    logger.info(f"Training {model_name.upper()} for {epochs} epochs")
    logger.info("=" * 60)

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (step + 1) % log_every == 0:
                logger.info(
                    f"  Epoch {epoch}, step {step + 1}: loss={loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation phase
        val_labels_np, val_probs_np = evaluate_gnn_epoch(model, val_loader, device)
        val_metrics = compute_multitask_metrics(
            val_labels_np, val_probs_np, task_names=TOX21_ASSAYS
        )

        val_pr_auc = val_metrics["mean"]["pr_auc"]
        val_roc_auc = val_metrics["mean"]["roc_auc"]
        current_lr = optimiser.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"val_PR-AUC={val_pr_auc:.4f}  "
            f"val_ROC-AUC={val_roc_auc:.4f}  "
            f"lr={current_lr:.6f}"
        )

        # LR scheduler step
        scheduler.step(val_pr_auc)

        # Early stopping check
        if val_pr_auc > best_val_metric:
            best_val_metric = val_pr_auc
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            if save_checkpoints:
                checkpoint_path = models_dir / f"{model_name}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "val_pr_auc": val_pr_auc,
                        "val_roc_auc": val_roc_auc,
                        "config": config,
                    },
                    checkpoint_path,
                )
                logger.info(f"  New best model saved (val PR-AUC={val_pr_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= es_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {es_patience} epochs)"
                )
                break

    train_time = time.time() - t0
    logger.info(f"\nTraining completed in {train_time:.1f}s")
    logger.info(f"Best epoch: {best_epoch} (val PR-AUC={best_val_metric:.4f})")

    # Load best model and evaluate
    checkpoint_path = models_dir / f"{model_name}_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")

    # Final validation metrics
    val_labels_np, val_probs_np = evaluate_gnn_epoch(model, val_loader, device)
    val_metrics = compute_multitask_metrics(
        val_labels_np, val_probs_np, task_names=TOX21_ASSAYS
    )

    logger.info(f"\n{model_name.upper()} best model validation results:")
    _log_metrics(val_metrics)

    # Tune thresholds on validation set
    thresholds = {}
    for i, task_name in enumerate(TOX21_ASSAYS):
        thresh, score = find_optimal_threshold(
            val_labels_np[:, i], val_probs_np[:, i], metric="f1"
        )
        thresholds[task_name] = thresh

    # Save metrics and thresholds
    _save_metrics_csv(
        val_metrics, results_dir / f"{model_name}_val_metrics.csv", model_name.upper()
    )
    _save_thresholds(thresholds, results_dir / f"{model_name}_thresholds.csv")

    logger.info("=" * 60)
    logger.info(f"{model_name.upper()} TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Best val PR-AUC: {val_metrics['mean']['pr_auc']:.4f}, "
        f"ROC-AUC: {val_metrics['mean']['roc_auc']:.4f} "
        f"(epoch {best_epoch}, {train_time:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Baseline training (unchanged)
# ---------------------------------------------------------------------------

def train_baselines(config: dict) -> None:
    """Train Random Forest and XGBoost baseline models.

    Args:
        config: Configuration dictionary from YAML file.
    """
    data_cfg = config.get("data", {})
    fp_radius = data_cfg.get("fingerprint_radius", 2)
    fp_bits = data_cfg.get("fingerprint_bits", 2048)

    # Load splits
    logger.info("Loading data splits...")
    _, train_smiles, train_labels = load_split_data("train")
    _, val_smiles, val_labels = load_split_data("val")

    # Featurise
    logger.info("Featurising with Morgan fingerprints...")
    X_train, y_train = featurise_fingerprints(
        train_smiles, train_labels, radius=fp_radius, n_bits=fp_bits
    )
    X_val, y_val = featurise_fingerprints(
        val_smiles, val_labels, radius=fp_radius, n_bits=fp_bits
    )

    models_dir = OUTPUTS_DIR / "models"
    results_dir = OUTPUTS_DIR / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})

    # ---- Random Forest ----
    rf_params = config.get("random_forest", {})
    logger.info("=" * 60)
    logger.info("Training Random Forest baselines")
    logger.info("=" * 60)

    rf = PerTaskClassifier(
        model_type="random_forest",
        task_names=TOX21_ASSAYS,
        model_params=rf_params,
        auto_pos_weight=True,
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0
    logger.info(f"RF training time: {rf_train_time:.1f}s")

    # Evaluate on validation set
    rf_probs_val = rf.predict_proba(X_val)
    rf_val_metrics = compute_multitask_metrics(y_val, rf_probs_val, task_names=TOX21_ASSAYS)

    logger.info("Random Forest validation results:")
    _log_metrics(rf_val_metrics)

    # Tune thresholds on validation set
    rf_thresholds = {}
    if eval_cfg.get("threshold_tuning", True):
        threshold_metric = eval_cfg.get("threshold_metric", "f1")
        for i, task_name in enumerate(TOX21_ASSAYS):
            thresh, score = find_optimal_threshold(
                y_val[:, i], rf_probs_val[:, i], metric=threshold_metric
            )
            rf_thresholds[task_name] = thresh
            logger.info(f"  {task_name}: optimal threshold={thresh:.2f} ({threshold_metric}={score:.3f})")

    # Save RF model
    rf.save(models_dir / "random_forest.pkl")

    # ---- XGBoost ----
    xgb_params = config.get("xgboost", {}).copy()
    early_stopping = xgb_params.pop("early_stopping_rounds", 50)

    logger.info("=" * 60)
    logger.info("Training XGBoost baselines")
    logger.info("=" * 60)

    xgb = PerTaskClassifier(
        model_type="xgboost",
        task_names=TOX21_ASSAYS,
        model_params=xgb_params,
        auto_pos_weight=True,
    )

    t0 = time.time()
    xgb.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=early_stopping,
    )
    xgb_train_time = time.time() - t0
    logger.info(f"XGBoost training time: {xgb_train_time:.1f}s")

    # Evaluate on validation set
    xgb_probs_val = xgb.predict_proba(X_val)
    xgb_val_metrics = compute_multitask_metrics(
        y_val, xgb_probs_val, task_names=TOX21_ASSAYS
    )

    logger.info("XGBoost validation results:")
    _log_metrics(xgb_val_metrics)

    # Tune thresholds on validation set
    xgb_thresholds = {}
    if eval_cfg.get("threshold_tuning", True):
        threshold_metric = eval_cfg.get("threshold_metric", "f1")
        for i, task_name in enumerate(TOX21_ASSAYS):
            thresh, score = find_optimal_threshold(
                y_val[:, i], xgb_probs_val[:, i], metric=threshold_metric
            )
            xgb_thresholds[task_name] = thresh

    # Save XGBoost model
    xgb.save(models_dir / "xgboost.pkl")

    # Save validation metrics to CSV
    _save_metrics_csv(rf_val_metrics, results_dir / "rf_val_metrics.csv", "Random Forest")
    _save_metrics_csv(xgb_val_metrics, results_dir / "xgb_val_metrics.csv", "XGBoost")

    # Save thresholds
    _save_thresholds(rf_thresholds, results_dir / "rf_thresholds.csv")
    _save_thresholds(xgb_thresholds, results_dir / "xgb_thresholds.csv")

    # Summary
    logger.info("=" * 60)
    logger.info("BASELINE TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"Random Forest  — val mean PR-AUC: "
        f"{rf_val_metrics['mean']['pr_auc']:.4f}, "
        f"val mean ROC-AUC: {rf_val_metrics['mean']['roc_auc']:.4f} "
        f"({rf_train_time:.1f}s)"
    )
    logger.info(
        f"XGBoost        — val mean PR-AUC: "
        f"{xgb_val_metrics['mean']['pr_auc']:.4f}, "
        f"val mean ROC-AUC: {xgb_val_metrics['mean']['roc_auc']:.4f} "
        f"({xgb_train_time:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_metrics(metrics: dict) -> None:
    """Log per-task metrics."""
    for task_name, task_metrics in metrics.items():
        if task_name == "mean":
            continue
        logger.info(
            f"  {task_name:15s}  PR-AUC={task_metrics['pr_auc']:.4f}  "
            f"ROC-AUC={task_metrics['roc_auc']:.4f}"
        )
    logger.info(
        f"  {'MEAN':15s}  PR-AUC={metrics['mean']['pr_auc']:.4f}  "
        f"ROC-AUC={metrics['mean']['roc_auc']:.4f}"
    )


def _save_metrics_csv(metrics: dict, path: Path, model_name: str) -> None:
    """Save metrics dictionary to CSV."""
    rows = []
    for task_name, task_metrics in metrics.items():
        row = {"model": model_name, "task": task_name}
        row.update(task_metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info(f"Saved metrics to {path}")


def _save_thresholds(thresholds: dict, path: Path) -> None:
    """Save optimal thresholds to CSV."""
    df = pd.DataFrame(
        [{"task": k, "threshold": v} for k, v in thresholds.items()]
    )
    df.to_csv(path, index=False)
    logger.info(f"Saved thresholds to {path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train toxicity prediction models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Route to appropriate training function
    if "random_forest" in config or "xgboost" in config:
        train_baselines(config)
    elif "model" in config:
        train_gnn(config)
    else:
        raise ValueError(
            "Config does not contain recognised model settings. "
            "Expected 'random_forest'/'xgboost' for baselines, "
            "or 'model' for GNNs."
        )


if __name__ == "__main__":
    main()
