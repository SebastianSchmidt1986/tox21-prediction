"""Scaffold-based dataset splitting for molecular data."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.chem import get_murcko_scaffold

logger = logging.getLogger(__name__)


def generate_scaffolds(
    smiles_list: list[str],
    generic: bool = False,
) -> dict[str, list[int]]:
    """Generate scaffold to molecule index mapping.

    Args:
        smiles_list: List of SMILES strings.
        generic: If True, use generic scaffolds (all atoms as carbon).

    Returns:
        Dictionary mapping scaffold SMILES to list of molecule indices.
    """
    scaffold_to_indices = defaultdict(list)

    for i, smiles in enumerate(smiles_list):
        scaffold = get_murcko_scaffold(smiles, generic=generic)
        if scaffold is None:
            # Use empty string for molecules without scaffold
            scaffold = ""
        scaffold_to_indices[scaffold].append(i)

    return dict(scaffold_to_indices)


def scaffold_split(
    smiles_list: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = 42,
    generic_scaffolds: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Split molecules based on Murcko scaffolds.

    Ensures that molecules with the same scaffold are in the same split,
    preventing data leakage from structural similarity.

    Args:
        smiles_list: List of SMILES strings.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        generic_scaffolds: If True, use generic scaffolds.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Generate scaffolds
    logger.info("Generating molecular scaffolds...")
    scaffold_to_indices = generate_scaffolds(smiles_list, generic=generic_scaffolds)
    logger.info(f"Found {len(scaffold_to_indices)} unique scaffolds")

    # Sort scaffolds by size (largest first) for balanced distribution
    scaffold_sets = list(scaffold_to_indices.values())
    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(scaffold_sets)

    # Sort by size descending to put larger scaffolds in training
    scaffold_sets.sort(key=len, reverse=True)

    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold_indices in scaffold_sets:
        if len(train_indices) < n_train:
            train_indices.extend(scaffold_indices)
        elif len(val_indices) < n_val:
            val_indices.extend(scaffold_indices)
        else:
            test_indices.extend(scaffold_indices)

    logger.info(
        f"Split sizes: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    return train_indices, val_indices, test_indices


def verify_no_scaffold_leakage(
    smiles_list: list[str],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> bool:
    """Verify that no scaffolds are shared between splits.

    Args:
        smiles_list: List of SMILES strings.
        train_indices: Training set indices.
        val_indices: Validation set indices.
        test_indices: Test set indices.

    Returns:
        True if no leakage detected, False otherwise.
    """
    train_scaffolds = set()
    for i in train_indices:
        scaffold = get_murcko_scaffold(smiles_list[i])
        if scaffold:
            train_scaffolds.add(scaffold)

    val_scaffolds = set()
    for i in val_indices:
        scaffold = get_murcko_scaffold(smiles_list[i])
        if scaffold:
            val_scaffolds.add(scaffold)

    test_scaffolds = set()
    for i in test_indices:
        scaffold = get_murcko_scaffold(smiles_list[i])
        if scaffold:
            test_scaffolds.add(scaffold)

    # Check for overlaps
    train_val_overlap = train_scaffolds & val_scaffolds
    train_test_overlap = train_scaffolds & test_scaffolds
    val_test_overlap = val_scaffolds & test_scaffolds

    if train_val_overlap:
        logger.warning(f"Train-val scaffold overlap: {len(train_val_overlap)} scaffolds")
        return False
    if train_test_overlap:
        logger.warning(f"Train-test scaffold overlap: {len(train_test_overlap)} scaffolds")
        return False
    if val_test_overlap:
        logger.warning(f"Val-test scaffold overlap: {len(val_test_overlap)} scaffolds")
        return False

    logger.info("No scaffold leakage detected between splits")
    return True


def create_split_dataframes(
    df: pd.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create separate DataFrames for each split.

    Args:
        df: Full DataFrame.
        train_indices: Training set indices.
        val_indices: Validation set indices.
        test_indices: Test set indices.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, val_df, test_df


def log_split_statistics(
    df: pd.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    label_cols: list[str],
) -> None:
    """Log statistics about the split.

    Args:
        df: Full DataFrame.
        train_indices: Training set indices.
        val_indices: Validation set indices.
        test_indices: Test set indices.
        label_cols: List of label column names.
    """
    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    logger.info("\nSplit statistics:")
    for split_name, indices in splits.items():
        split_df = df.iloc[indices]
        logger.info(f"\n{split_name.upper()} ({len(indices)} samples):")

        for col in label_cols:
            if col in split_df.columns:
                n_valid = split_df[col].notna().sum()
                n_pos = (split_df[col] == 1).sum()
                pos_rate = n_pos / n_valid if n_valid > 0 else 0
                logger.info(f"  {col}: {n_pos}/{n_valid} positive ({pos_rate:.1%})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.load import TOX21_ASSAYS

    data_dir = Path(__file__).parent.parent.parent / "data"
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    input_path = processed_dir / "tox21_sanitised.csv"

    if not input_path.exists():
        print(f"Sanitised data not found at {input_path}")
        print("Run src/data/sanitise.py first")
        exit(1)

    # Load data
    df = pd.read_csv(input_path)
    smiles_list = df["smiles"].tolist()
    print(f"Loaded {len(df)} molecules")

    # Perform scaffold split
    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    # Verify no leakage
    verify_no_scaffold_leakage(smiles_list, train_idx, val_idx, test_idx)

    # Log statistics
    log_split_statistics(df, train_idx, val_idx, test_idx, TOX21_ASSAYS)

    # Create split DataFrames
    train_df, val_df, test_df = create_split_dataframes(
        df, train_idx, val_idx, test_idx
    )

    # Save splits
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # Save indices for reproducibility
    np.save(splits_dir / "train_indices.npy", np.array(train_idx))
    np.save(splits_dir / "val_indices.npy", np.array(val_idx))
    np.save(splits_dir / "test_indices.npy", np.array(test_idx))

    print(f"\nSaved splits to {splits_dir}")
    print(f"  train.csv: {len(train_df)} samples")
    print(f"  val.csv: {len(val_df)} samples")
    print(f"  test.csv: {len(test_df)} samples")
