"""RDKit chemistry utilities for molecular processing."""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES string to RDKit Mol object.

    Args:
        smiles: SMILES string representation of a molecule.

    Returns:
        RDKit Mol object if parsing succeeds, None otherwise.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        logger.warning(f"Failed to parse SMILES '{smiles}': {e}")
        return None


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> Optional[str]:
    """Convert RDKit Mol object to SMILES string.

    Args:
        mol: RDKit Mol object.
        canonical: Whether to return canonical SMILES.

    Returns:
        SMILES string if conversion succeeds, None otherwise.
    """
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=canonical)
    except Exception as e:
        logger.warning(f"Failed to convert mol to SMILES: {e}")
        return None


def get_murcko_scaffold(smiles: str, generic: bool = False) -> Optional[str]:
    """Extract Murcko scaffold from a SMILES string.

    The Murcko scaffold is the core ring structure of a molecule,
    used for scaffold-based dataset splitting to avoid data leakage.

    Args:
        smiles: SMILES string of the molecule.
        generic: If True, return generic scaffold (all atoms as carbon,
                 all bonds as single). Default False preserves atom types.

    Returns:
        SMILES string of the scaffold, or None if extraction fails.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    try:
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(
                MurckoScaffold.GetScaffoldForMol(mol)
            )
        else:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception as e:
        logger.warning(f"Failed to extract scaffold from '{smiles}': {e}")
        return None


def is_valid_mol(mol: Optional[Chem.Mol]) -> bool:
    """Check if an RDKit Mol object is valid.

    Args:
        mol: RDKit Mol object or None.

    Returns:
        True if mol is valid and not None.
    """
    if mol is None:
        return False
    try:
        # Try to get SMILES as a validity check
        Chem.MolToSmiles(mol)
        return True
    except Exception:
        return False


def count_atoms(mol: Chem.Mol) -> int:
    """Count the number of atoms in a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Number of atoms (excluding hydrogens unless explicit).
    """
    if mol is None:
        return 0
    return mol.GetNumAtoms()


def count_bonds(mol: Chem.Mol) -> int:
    """Count the number of bonds in a molecule.

    Args:
        mol: RDKit Mol object.

    Returns:
        Number of bonds.
    """
    if mol is None:
        return 0
    return mol.GetNumBonds()
