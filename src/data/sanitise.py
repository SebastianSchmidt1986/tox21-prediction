"""SMILES sanitisation and standardisation using RDKit."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


def standardise_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Standardise an RDKit molecule.

    Performs the following standardisation steps:
    1. Remove salts and solvents (keep largest fragment)
    2. Neutralise charges where possible
    3. Standardise tautomers
    4. Remove explicit hydrogens

    Args:
        mol: RDKit Mol object.

    Returns:
        Standardised Mol object, or None if standardisation fails.
    """
    if mol is None:
        return None

    try:
        # Remove salts/solvents - keep largest fragment
        mol = rdMolStandardize.FragmentParent(mol)

        # Neutralise charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Normalise functional groups
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        # Remove explicit hydrogens
        mol = Chem.RemoveHs(mol)

        # Sanitise to ensure validity
        Chem.SanitizeMol(mol)

        return mol

    except Exception as e:
        logger.debug(f"Standardisation failed: {e}")
        return None


def sanitise_smiles(smiles: str) -> Optional[str]:
    """Sanitise and standardise a SMILES string.

    Converts SMILES to molecule, standardises it, and returns
    the canonical SMILES representation.

    Args:
        smiles: Input SMILES string.

    Returns:
        Standardised canonical SMILES, or None if sanitisation fails.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"Failed to parse SMILES: {smiles}")
            return None

        # Standardise molecule
        mol = standardise_mol(mol)
        if mol is None:
            return None

        # Convert back to canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True)

    except Exception as e:
        logger.debug(f"Failed to sanitise SMILES '{smiles}': {e}")
        return None


def sanitise_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """Sanitise all SMILES in a DataFrame.

    Args:
        df: DataFrame with a SMILES column.
        smiles_col: Name of the SMILES column.
        drop_invalid: If True, drop rows with invalid SMILES.
                     If False, keep them with NaN in sanitised column.

    Returns:
        DataFrame with additional 'smiles_sanitised' column
        (or with invalid rows dropped).
    """
    logger.info(f"Sanitising {len(df)} molecules...")

    df = df.copy()

    # Apply sanitisation
    df["smiles_sanitised"] = df[smiles_col].apply(sanitise_smiles)

    # Count results
    n_valid = df["smiles_sanitised"].notna().sum()
    n_invalid = len(df) - n_valid
    logger.info(f"Sanitisation complete: {n_valid} valid, {n_invalid} invalid")

    if drop_invalid:
        df = df[df["smiles_sanitised"].notna()].copy()
        logger.info(f"Dropped invalid molecules. {len(df)} remaining.")

    # Check for duplicates after sanitisation
    n_duplicates = df["smiles_sanitised"].duplicated().sum()
    if n_duplicates > 0:
        logger.warning(
            f"Found {n_duplicates} duplicate SMILES after sanitisation. "
            "Consider deduplication."
        )

    return df


def deduplicate_smiles(
    df: pd.DataFrame,
    smiles_col: str = "smiles_sanitised",
    label_cols: Optional[list[str]] = None,
    strategy: str = "first",
) -> pd.DataFrame:
    """Deduplicate DataFrame by SMILES, handling label conflicts.

    Args:
        df: DataFrame with SMILES column.
        smiles_col: Name of the SMILES column to deduplicate on.
        label_cols: List of label columns. If provided and strategy='vote',
                    conflicting labels are resolved by majority vote.
        strategy: 'first' keeps first occurrence, 'vote' uses majority vote
                  for labels (only with label_cols).

    Returns:
        Deduplicated DataFrame.
    """
    if strategy == "first" or label_cols is None:
        return df.drop_duplicates(subset=[smiles_col], keep="first")

    elif strategy == "vote":
        # Group by SMILES and take majority vote for each label
        result_rows = []
        for smiles, group in df.groupby(smiles_col):
            row = group.iloc[0].copy()
            for col in label_cols:
                if col in group.columns:
                    # Take majority vote, treating NaN as abstention
                    valid_labels = group[col].dropna()
                    if len(valid_labels) > 0:
                        row[col] = valid_labels.mode().iloc[0]
            result_rows.append(row)

        return pd.DataFrame(result_rows)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.load import TOX21_ASSAYS, load_tox21_sdf

    data_dir = Path(__file__).parent.parent.parent / "data"
    sdf_path = data_dir / "raw" / "tox21_10k_data_all.sdf"

    if not sdf_path.exists():
        print(f"SDF file not found at {sdf_path}")
        exit(1)

    # Load raw data
    df = load_tox21_sdf(sdf_path)
    print(f"Loaded {len(df)} molecules")

    # Sanitise
    df = sanitise_dataframe(df, drop_invalid=True)
    print(f"After sanitisation: {len(df)} molecules")

    # Deduplicate
    df = deduplicate_smiles(df, label_cols=TOX21_ASSAYS, strategy="first")
    print(f"After deduplication: {len(df)} molecules")

    # Use sanitised SMILES as main SMILES column
    df["smiles_original"] = df["smiles"]
    df["smiles"] = df["smiles_sanitised"]
    df = df.drop(columns=["smiles_sanitised"])

    # Save
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    output_path = processed_dir / "tox21_sanitised.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved sanitised data to {output_path}")

    # Print final statistics
    print("\nFinal dataset statistics:")
    print(f"  Total compounds: {len(df)}")
    for assay in TOX21_ASSAYS:
        if assay in df.columns:
            n_valid = df[assay].notna().sum()
            n_pos = (df[assay] == 1).sum()
            print(f"  {assay}: {n_pos}/{n_valid} positive ({n_pos/n_valid:.1%})")
