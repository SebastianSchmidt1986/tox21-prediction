"""GNN explanations using integrated gradients (Captum)."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem import Draw

from src.data.featurise import smiles_to_pyg_data
from src.data.load import TOX21_ASSAYS
from src.models.gnn import build_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


class GNNForwardWrapper(torch.nn.Module):
    """Wrapper that makes a GNN compatible with Captum's attribution API.

    Captum expects a forward function that takes only the input features
    being attributed. This wrapper fixes edge_index and batch, attributing
    only the node features.

    Args:
        model: The GNN model.
        edge_index: Fixed edge index tensor.
        batch: Fixed batch tensor.
        task_idx: Index of the task (assay) to explain.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        task_idx: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.batch = batch
        self.task_idx = task_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the logit for a single task.

        Captum may batch multiple interpolation steps into a single call,
        stacking N copies of the node features along dim 0. This method
        replicates edge_index and batch accordingly.

        Args:
            x: Node features, shape (num_atoms * n_copies, num_features).

        Returns:
            Logits for the target task, shape (n_copies,).
        """
        n_atoms = self.batch.shape[0]
        n_copies = x.shape[0] // n_atoms

        if n_copies > 1:
            # Replicate graph structure for each interpolation step
            offsets = torch.arange(
                n_copies, device=x.device, dtype=torch.long
            ) * n_atoms
            edge_index = torch.cat(
                [self.edge_index + off for off in offsets], dim=1
            )
            batch = torch.cat(
                [self.batch + i for i in range(n_copies)]
            )
        else:
            edge_index = self.edge_index
            batch = self.batch

        logits = self.model(x, edge_index, batch)
        return logits[:, self.task_idx]


def compute_atom_attributions(
    model: torch.nn.Module,
    smiles: str,
    task_idx: int,
    device: torch.device,
    n_steps: int = 50,
) -> tuple[np.ndarray | None, str]:
    """Compute per-atom attributions using integrated gradients.

    Args:
        model: GNN model.
        smiles: SMILES string.
        task_idx: Assay index to explain.
        device: Torch device.
        n_steps: Number of interpolation steps for IG.

    Returns:
        Tuple of (atom_attributions, smiles). atom_attributions has shape
        (num_atoms,), or None if conversion fails.
    """
    data = smiles_to_pyg_data(smiles)
    if data is None:
        return None, smiles

    data = data.to(device)
    # Single-molecule batch
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    wrapper = GNNForwardWrapper(model, data.edge_index, batch, task_idx)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)

    # Baseline: zero node features
    input_x = data.x.clone().detach().float().requires_grad_(True)
    baseline = torch.zeros_like(input_x)

    attributions = ig.attribute(
        input_x,
        baselines=baseline,
        n_steps=n_steps,
    )

    # Sum attributions across feature dimensions to get per-atom importance
    atom_attr = attributions.detach().cpu().numpy()
    atom_importance = np.abs(atom_attr).sum(axis=1)

    return atom_importance, smiles


def explain_gnn(
    checkpoint_path: Path,
    assays: list[str] | None = None,
    n_molecules: int = 20,
) -> dict[str, list[dict]]:
    """Compute atom-level attributions for a GNN model.

    For each assay, explains the top predicted-positive molecules from
    the test set using integrated gradients.

    Args:
        checkpoint_path: Path to GNN checkpoint.
        assays: Assays to explain. Defaults to 3 representative ones.
        n_molecules: Number of molecules to explain per assay.

    Returns:
        Dictionary mapping assay names to lists of attribution results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_name = config["model"]["name"]

    logger.info(f"Loaded {model_name.upper()} from epoch {checkpoint['epoch']}")

    if assays is None:
        assays = ["NR-AhR", "SR-MMP", "SR-ARE"]

    # Load test data
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_smiles = test_df["smiles"].tolist()
    test_labels = test_df[TOX21_ASSAYS].values.astype(np.float32)

    # Build model — need in_dim from a sample molecule
    sample_data = smiles_to_pyg_data(test_smiles[0])
    in_dim = sample_data.x.shape[1]
    model = build_model(config, in_dim=in_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get predictions for all test molecules to find top positives
    from src.training.train import build_pyg_dataset, evaluate_gnn_epoch
    from torch_geometric.loader import DataLoader

    test_dataset = build_pyg_dataset(test_smiles, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    _, test_probs = evaluate_gnn_epoch(model, test_loader, device)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for assay in assays:
        task_idx = TOX21_ASSAYS.index(assay)
        logger.info(f"\nComputing atom attributions for {assay} ({model_name})...")

        # Select top predicted-positive molecules with observed labels
        probs = test_probs[:, task_idx]
        labels = test_labels[:, task_idx]
        observed_mask = ~np.isnan(labels)

        # Sort by predicted probability (descending)
        sorted_idx = np.argsort(probs)[::-1]
        selected = []
        for idx in sorted_idx:
            if observed_mask[idx] and len(selected) < n_molecules:
                selected.append(idx)

        assay_results = []
        atom_attr_list = []

        for mol_idx in selected:
            smiles = test_smiles[mol_idx]
            attr, _ = compute_atom_attributions(
                model, smiles, task_idx, device, n_steps=50
            )
            if attr is not None:
                true_label = int(labels[mol_idx])
                pred_prob = float(probs[mol_idx])

                result = {
                    "assay": assay,
                    "smiles": smiles,
                    "true_label": true_label,
                    "pred_prob": pred_prob,
                    "num_atoms": len(attr),
                    "max_attribution": float(attr.max()),
                    "top_atom_idx": int(np.argmax(attr)),
                }
                assay_results.append(result)
                atom_attr_list.append((smiles, attr, true_label, pred_prob))

        all_results[assay] = assay_results

        # Plot attributions for top 5 molecules
        n_plot = min(5, len(atom_attr_list))
        if n_plot > 0:
            fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4))
            if n_plot == 1:
                axes = [axes]

            for i, (smiles, attr, true_label, pred_prob) in enumerate(
                atom_attr_list[:n_plot]
            ):
                ax = axes[i]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Normalise attributions to [0, 1]
                attr_norm = attr / (attr.max() + 1e-8)

                ax.bar(range(len(attr_norm)), attr_norm, color="coral")
                ax.set_xlabel("Atom index")
                ax.set_ylabel("Attribution")
                from rdkit.Chem import rdMolDescriptors
                formula = rdMolDescriptors.CalcMolFormula(mol)
                formula_sub = formula.translate(
                    str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                )
                ax.set_title(
                    f"y={true_label}  p={pred_prob:.2f}\n{formula_sub}",
                    fontsize=9,
                )

            plt.suptitle(
                f"Atom attributions — {assay} ({model_name.upper()})", fontsize=12
            )
            plt.tight_layout()

            save_path = (
                FIGURES_DIR / f"gnn_attr_{model_name}_{assay.replace('-', '_')}.png"
            )
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved attribution plot to {save_path}")

        # Log summary
        if assay_results:
            logger.info(f"  Explained {len(assay_results)} molecules for {assay}")
            for r in assay_results[:5]:
                logger.info(
                    f"    {r['smiles'][:40]:40s}  "
                    f"y={r['true_label']}  p={r['pred_prob']:.3f}  "
                    f"top_atom={r['top_atom_idx']}  max_attr={r['max_attribution']:.4f}"
                )

    return all_results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GNN atom-level explanations")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/gcn_best.pt",
        help="Path to GNN checkpoint",
    )
    parser.add_argument(
        "--assays",
        type=str,
        nargs="+",
        default=None,
        help="Assays to explain (default: 3 representative)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    checkpoint_path = Path(args.model)
    results = explain_gnn(checkpoint_path, assays=args.assays)

    # Save results
    results_dir = OUTPUTS_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for assay, assay_results in results.items():
        all_rows.extend(assay_results)

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_path = results_dir / "gnn_atom_attributions.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved attribution results to {output_path}")


if __name__ == "__main__":
    main()
