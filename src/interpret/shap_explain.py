"""SHAP explanations for fingerprint-based baseline models."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from src.data.featurise import compute_morgan_fingerprints
from src.data.load import TOX21_ASSAYS
from src.models.baseline import PerTaskClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def get_bit_info_for_molecule(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[int, list]:
    """Get Morgan fingerprint bit information for a molecule.

    Maps each active bit to the atoms and radius that set it.

    Args:
        smiles: SMILES string.
        radius: Fingerprint radius.
        n_bits: Fingerprint length.

    Returns:
        Dictionary mapping bit index to list of (atom_idx, radius) tuples.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )
    return bit_info


def explain_baseline(
    model_path: Path,
    assays: list[str] | None = None,
    n_background: int = 200,
    n_explain: int = 100,
) -> dict[str, np.ndarray]:
    """Compute SHAP values for a baseline model using TreeExplainer.

    Args:
        model_path: Path to saved PerTaskClassifier.
        assays: List of assay names to explain. If None, explains all.
        n_background: Number of background samples for the explainer.
        n_explain: Number of test samples to explain.

    Returns:
        Dictionary mapping assay names to SHAP value arrays.
    """
    model = PerTaskClassifier.load(model_path)
    model_name = model.model_type

    if assays is None:
        assays = TOX21_ASSAYS

    # Load test data
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_smiles = test_df["smiles"].tolist()
    test_labels = test_df[TOX21_ASSAYS].values.astype(np.float32)

    X_test, valid_indices = compute_morgan_fingerprints(test_smiles)
    y_test = test_labels[valid_indices]
    valid_smiles = [test_smiles[i] for i in valid_indices]

    # Use a subset for background and explanation
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_test), min(n_background, len(X_test)), replace=False)
    explain_idx = np.random.choice(len(X_test), min(n_explain, len(X_test)), replace=False)

    X_background = X_test[bg_idx]
    X_explain = X_test[explain_idx]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_shap_values = {}

    for assay in assays:
        if assay not in model.models:
            logger.warning(f"No model for {assay}, skipping")
            continue

        task_model = model.models[assay]
        logger.info(f"Computing SHAP values for {assay} ({model_name})...")

        explainer = shap.TreeExplainer(task_model, data=X_background)
        shap_values = explainer.shap_values(X_explain)

        # Handle different return formats from TreeExplainer.
        # May return a list [class_0, class_1] or an ndarray with extra dims.
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[1])  # Take positive class
        else:
            shap_values = np.array(shap_values)

        # Ensure 2D (n_samples, n_features) — collapse any trailing class dims
        while shap_values.ndim > 2:
            shap_values = shap_values[..., -1]

        all_shap_values[assay] = shap_values

        # Summary plot (bar) — top 20 features
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-20:][::-1]
        top_bits = [f"bit_{i}" for i in top_idx]
        top_values = mean_abs_shap[top_idx]

        ax.barh(range(len(top_idx)), top_values[::-1], color="steelblue")
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(top_bits[::-1], fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        display_name = model_name.replace("_", " ").title()
        ax.set_title(f"Top 20 fingerprint bits — {assay} ({display_name})")
        plt.tight_layout()

        save_path = FIGURES_DIR / f"shap_{model_name}_{assay.replace('-', '_')}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved SHAP plot to {save_path}")

        # Log top 10 bits
        logger.info(f"  Top 10 bits for {assay}:")
        for rank, idx in enumerate(top_idx[:10]):
            logger.info(f"    {rank + 1}. bit_{idx}: mean|SHAP|={mean_abs_shap[idx]:.4f}")

    return all_shap_values


def explain_top_bits(
    shap_values: np.ndarray,
    smiles_list: list[str],
    assay_name: str,
    top_k: int = 5,
    radius: int = 2,
    n_bits: int = 2048,
) -> list[dict]:
    """Identify molecular substructures corresponding to top SHAP bits.

    For the top-k most important fingerprint bits, find example molecules
    and the atom environments that activate those bits.

    Args:
        shap_values: SHAP values, shape (n_samples, n_bits).
        smiles_list: SMILES for the explained samples.
        assay_name: Name of the assay.
        top_k: Number of top bits to analyse.
        radius: Morgan fingerprint radius.
        n_bits: Fingerprint length.

    Returns:
        List of dictionaries with bit info.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]

    results = []
    for bit_idx in top_idx:
        # Find a molecule where this bit is active
        example_smiles = None
        example_atoms = None

        for smiles in smiles_list:
            bit_info = get_bit_info_for_molecule(smiles, radius=radius, n_bits=n_bits)
            if bit_idx in bit_info:
                example_smiles = smiles
                example_atoms = bit_info[bit_idx]
                break

        results.append({
            "assay": assay_name,
            "bit": bit_idx,
            "mean_abs_shap": float(mean_abs_shap[bit_idx]),
            "mean_shap": float(shap_values[:, bit_idx].mean()),
            "example_smiles": example_smiles,
            "example_atoms": example_atoms,
        })

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SHAP explanations for baselines")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/random_forest.pkl",
        help="Path to saved baseline model",
    )
    parser.add_argument(
        "--assays",
        type=str,
        nargs="+",
        default=None,
        help="Assays to explain (default: 3 representative assays)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_path = Path(args.model)

    # Default: explain 3 representative assays (best, medium, worst for RF)
    assays = args.assays or ["NR-AhR", "SR-MMP", "SR-ARE"]

    shap_values = explain_baseline(model_path, assays=assays)

    # Analyse top bits for each assay
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_smiles = test_df["smiles"].tolist()

    results_dir = OUTPUTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_bit_info = []
    for assay, sv in shap_values.items():
        bits = explain_top_bits(sv, test_smiles, assay, top_k=10)
        all_bit_info.extend(bits)

        logger.info(f"\nTop substructure bits for {assay}:")
        for b in bits[:5]:
            direction = "toxic" if b["mean_shap"] > 0 else "non-toxic"
            logger.info(
                f"  bit_{b['bit']}: |SHAP|={b['mean_abs_shap']:.4f} "
                f"(pushes {direction}), example: {b['example_smiles']}"
            )

    # Save bit analysis
    bit_df = pd.DataFrame(all_bit_info)
    bit_path = results_dir / "shap_top_bits.csv"
    bit_df.to_csv(bit_path, index=False)
    logger.info(f"\nSaved top bit analysis to {bit_path}")


if __name__ == "__main__":
    main()
