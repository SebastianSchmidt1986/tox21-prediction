"""Load Tox21 dataset from SDF files."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)

# Tox21 assay names as they appear in the SDF file
TOX21_ASSAYS = [
    "NR-AhR",
    "NR-AR",
    "NR-AR-LBD",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "NR-Aromatase",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

# Mapping from SDF property names to standardised names
ASSAY_NAME_MAP = {
    "NR-PPAR-gamma": "NR-PPARg",  # Standardise name
}


def load_tox21_sdf(
    sdf_path: str | Path,
    assays: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load Tox21 dataset from SDF file.

    Extracts SMILES representations and assay labels from the SDF file.
    Missing assay labels are represented as NaN (not 0).

    Args:
        sdf_path: Path to the Tox21 SDF file.
        assays: List of assay names to extract. If None, extracts all 12 assays.

    Returns:
        DataFrame with columns: 'smiles', 'mol_id', and one column per assay.
        Assay columns contain 0/1 for labels, NaN for missing.
    """
    sdf_path = Path(sdf_path)
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    if assays is None:
        assays = TOX21_ASSAYS

    logger.info(f"Loading Tox21 data from {sdf_path}")

    # Read SDF file
    suppl = Chem.SDMolSupplier(str(sdf_path))

    records = []
    failed_count = 0
    total_count = 0

    for mol in suppl:
        total_count += 1

        if mol is None:
            failed_count += 1
            continue

        # Extract SMILES
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.debug(f"Failed to generate SMILES: {e}")
            failed_count += 1
            continue

        # Extract molecule ID
        props = mol.GetPropsAsDict()
        mol_id = props.get("DSSTox_CID", f"mol_{total_count}")

        # Extract assay labels
        record = {
            "smiles": smiles,
            "mol_id": mol_id,
        }

        for assay in assays:
            if assay in props:
                try:
                    value = int(props[assay])
                    record[assay] = value
                except (ValueError, TypeError):
                    record[assay] = np.nan
            else:
                record[assay] = np.nan

        records.append(record)

    df = pd.DataFrame(records)

    # Log statistics
    n_compounds = len(df)
    logger.info(f"Loaded {n_compounds} compounds ({failed_count} failed to parse)")

    # Log per-assay statistics
    logger.info("Per-assay label distribution:")
    for assay in assays:
        if assay in df.columns:
            valid = df[assay].notna()
            n_valid = valid.sum()
            n_positive = (df[assay] == 1).sum()
            n_negative = (df[assay] == 0).sum()
            missing_rate = 1 - (n_valid / len(df))
            pos_rate = n_positive / n_valid if n_valid > 0 else 0
            logger.info(
                f"  {assay}: {n_positive} pos, {n_negative} neg, "
                f"{missing_rate:.1%} missing, {pos_rate:.1%} active rate"
            )

    return df


def load_tox21_from_deepchem() -> pd.DataFrame:
    """Load Tox21 dataset using DeepChem's data loader.

    This is an alternative loader that uses DeepChem's curated version
    of the Tox21 dataset with pre-computed splits.

    Returns:
        DataFrame with SMILES and assay labels.
    """
    try:
        import deepchem as dc
    except ImportError:
        raise ImportError(
            "DeepChem is required for this loader. Install with: pip install deepchem"
        )

    logger.info("Loading Tox21 from DeepChem")
    _, datasets, _ = dc.molnet.load_tox21(featurizer="Raw")

    # Combine all splits
    all_smiles = []
    all_labels = []
    for dataset in datasets:
        all_smiles.extend(dataset.ids)
        all_labels.append(dataset.y)

    labels = np.vstack(all_labels)

    # Create DataFrame
    df = pd.DataFrame({"smiles": all_smiles})
    for i, assay in enumerate(TOX21_ASSAYS):
        # DeepChem uses -1 for missing labels
        col = labels[:, i]
        col = np.where(col == -1, np.nan, col)
        df[assay] = col

    df["mol_id"] = [f"dc_{i}" for i in range(len(df))]

    logger.info(f"Loaded {len(df)} compounds from DeepChem")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load from SDF file
    data_dir = Path(__file__).parent.parent.parent / "data"
    sdf_path = data_dir / "raw" / "tox21_10k_data_all.sdf"

    if sdf_path.exists():
        df = load_tox21_sdf(sdf_path)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:\n{df.head()}")

        # Save to processed directory
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        output_path = processed_dir / "tox21_raw.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved raw data to {output_path}")
    else:
        print(f"SDF file not found at {sdf_path}")
