"""Molecular featurisation for fingerprints and graph construction."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Atom feature constants
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # Periodic table
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridisation": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": [0, 1, 2, 3, 4],
}

# Bond feature constants
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def compute_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """Compute Morgan fingerprint (ECFP) for a molecule.

    Args:
        smiles: SMILES string.
        radius: Fingerprint radius. radius=2 gives ECFP4.
        n_bits: Length of the bit vector.

    Returns:
        Binary fingerprint as numpy array, or None if computation fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)

    except Exception as e:
        logger.debug(f"Failed to compute fingerprint for '{smiles}': {e}")
        return None


def compute_morgan_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[int]]:
    """Compute Morgan fingerprints for a list of molecules.

    Args:
        smiles_list: List of SMILES strings.
        radius: Fingerprint radius.
        n_bits: Length of the bit vector.

    Returns:
        Tuple of (fingerprint_matrix, valid_indices).
        fingerprint_matrix has shape (n_valid, n_bits).
        valid_indices contains indices of successfully processed molecules.
    """
    fingerprints = []
    valid_indices = []

    for i, smiles in enumerate(smiles_list):
        fp = compute_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)

    if fingerprints:
        return np.vstack(fingerprints), valid_indices
    else:
        return np.array([]).reshape(0, n_bits), []


def one_hot_encode(value, choices: list) -> list[int]:
    """One-hot encode a value given a list of choices.

    Args:
        value: Value to encode.
        choices: List of valid choices.

    Returns:
        One-hot encoded list. If value not in choices, returns all zeros.
    """
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def get_atom_features(atom: Chem.Atom) -> list[float]:
    """Extract features from an atom.

    Features:
    - Atomic number (one-hot)
    - Degree (one-hot)
    - Formal charge (one-hot)
    - Hybridisation (one-hot)
    - Is aromatic (binary)
    - Number of hydrogens (one-hot)

    Args:
        atom: RDKit Atom object.

    Returns:
        List of atom features.
    """
    features = []

    # Atomic number
    features.extend(one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"]))

    # Degree
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_FEATURES["degree"]))

    # Formal charge
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"]))

    # Hybridisation
    features.extend(one_hot_encode(atom.GetHybridization(), ATOM_FEATURES["hybridisation"]))

    # Is aromatic
    features.append(int(atom.GetIsAromatic()))

    # Number of Hs
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"]))

    return features


def get_bond_features(bond: Chem.Bond) -> list[float]:
    """Extract features from a bond.

    Features:
    - Bond type (one-hot)
    - Is conjugated (binary)
    - Is in ring (binary)

    Args:
        bond: RDKit Bond object.

    Returns:
        List of bond features.
    """
    features = []

    # Bond type
    features.extend(one_hot_encode(bond.GetBondType(), BOND_TYPES))

    # Is conjugated
    features.append(int(bond.GetIsConjugated()))

    # Is in ring
    features.append(int(bond.IsInRing()))

    return features


def mol_to_graph(smiles: str) -> Optional[dict]:
    """Convert a molecule to a graph representation.

    Creates a dictionary containing node features, edge indices,
    and edge features suitable for PyTorch Geometric.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary with keys:
        - 'x': Node features, shape (num_atoms, num_node_features)
        - 'edge_index': Edge indices, shape (2, num_edges)
        - 'edge_attr': Edge features, shape (num_edges, num_edge_features)
        - 'num_nodes': Number of nodes
        Returns None if conversion fails.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(get_atom_features(atom))

        # Get bond features and edge indices
        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_feat = get_bond_features(bond)

            # Add both directions (undirected graph)
            edge_indices.extend([[i, j], [j, i]])
            edge_features.extend([bond_feat, bond_feat])

        # Convert to numpy arrays
        x = np.array(atom_features, dtype=np.float32)

        if edge_indices:
            edge_index = np.array(edge_indices, dtype=np.int64).T
            edge_attr = np.array(edge_features, dtype=np.float32)
        else:
            # Handle molecules with no bonds (single atoms)
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 6), dtype=np.float32)  # 6 edge features

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": len(atom_features),
        }

    except Exception as e:
        logger.debug(f"Failed to convert '{smiles}' to graph: {e}")
        return None


def smiles_to_pyg_data(
    smiles: str,
    y: Optional[np.ndarray] = None,
):
    """Convert SMILES to PyTorch Geometric Data object.

    Args:
        smiles: SMILES string.
        y: Optional label array.

    Returns:
        PyTorch Geometric Data object, or None if conversion fails.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required. Install with: pip install torch-geometric"
        )

    graph_dict = mol_to_graph(smiles)
    if graph_dict is None:
        return None

    data = Data(
        x=torch.tensor(graph_dict["x"], dtype=torch.float),
        edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph_dict["edge_attr"], dtype=torch.float),
    )

    if y is not None:
        # Shape (1, num_tasks) so PyG batching concatenates to (batch, num_tasks)
        y_tensor = torch.tensor(y, dtype=torch.float)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(0)
        data.y = y_tensor

    return data


def featurise_dataset(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    label_cols: Optional[list[str]] = None,
    fp_radius: int = 2,
    fp_bits: int = 2048,
) -> dict:
    """Featurise a dataset with both fingerprints and graphs.

    Args:
        df: DataFrame with SMILES and labels.
        smiles_col: Name of SMILES column.
        label_cols: List of label column names.
        fp_radius: Morgan fingerprint radius.
        fp_bits: Morgan fingerprint bits.

    Returns:
        Dictionary containing:
        - 'fingerprints': numpy array of fingerprints
        - 'graphs': list of graph dictionaries
        - 'labels': numpy array of labels
        - 'valid_indices': indices of valid molecules
        - 'smiles': list of valid SMILES
    """
    smiles_list = df[smiles_col].tolist()

    # Compute fingerprints
    logger.info("Computing Morgan fingerprints...")
    fingerprints, valid_indices = compute_morgan_fingerprints(
        smiles_list, radius=fp_radius, n_bits=fp_bits
    )
    logger.info(f"Computed fingerprints for {len(valid_indices)} molecules")

    # Compute graphs
    logger.info("Computing molecular graphs...")
    graphs = []
    graph_indices = []
    for i in valid_indices:
        graph = mol_to_graph(smiles_list[i])
        if graph is not None:
            graphs.append(graph)
            graph_indices.append(i)

    # Use only molecules that have both fingerprints and graphs
    final_indices = graph_indices
    logger.info(f"Valid molecules with both representations: {len(final_indices)}")

    # Filter fingerprints to match graph indices
    fp_idx_map = {idx: i for i, idx in enumerate(valid_indices)}
    final_fingerprints = np.array([fingerprints[fp_idx_map[i]] for i in final_indices])

    # Extract labels
    if label_cols is not None:
        labels = df.iloc[final_indices][label_cols].values.astype(np.float32)
    else:
        labels = None

    # Get valid SMILES
    valid_smiles = [smiles_list[i] for i in final_indices]

    return {
        "fingerprints": final_fingerprints,
        "graphs": graphs,
        "labels": labels,
        "valid_indices": final_indices,
        "smiles": valid_smiles,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.data.load import TOX21_ASSAYS

    data_dir = Path(__file__).parent.parent.parent / "data"
    processed_dir = data_dir / "processed"
    input_path = processed_dir / "tox21_sanitised.csv"

    if not input_path.exists():
        print(f"Sanitised data not found at {input_path}")
        print("Run src/data/sanitise.py first")
        exit(1)

    # Load sanitised data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} molecules")

    # Featurise
    features = featurise_dataset(df, label_cols=TOX21_ASSAYS)

    print(f"\nFeaturisation results:")
    print(f"  Fingerprints shape: {features['fingerprints'].shape}")
    print(f"  Number of graphs: {len(features['graphs'])}")
    print(f"  Labels shape: {features['labels'].shape}")

    # Save fingerprints
    np.save(processed_dir / "fingerprints.npy", features["fingerprints"])
    np.save(processed_dir / "labels.npy", features["labels"])

    # Save SMILES list
    smiles_df = pd.DataFrame({"smiles": features["smiles"]})
    smiles_df.to_csv(processed_dir / "smiles.csv", index=False)

    print(f"\nSaved features to {processed_dir}")
