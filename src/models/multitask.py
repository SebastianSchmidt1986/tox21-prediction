"""Shared multi-task prediction head and masked BCE loss."""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):
    """MLP head that maps a graph-level representation to per-task predictions.

    Architecture: Linear → ReLU → Dropout → Linear (num_tasks outputs).

    Args:
        in_dim: Dimensionality of the input (pooled graph representation).
        hidden_dim: Hidden layer size.
        num_tasks: Number of output tasks (12 for Tox21).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_tasks: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Pooled graph representation, shape (batch_size, in_dim).

        Returns:
            Logits, shape (batch_size, num_tasks).
        """
        return self.head(x)


class MaskedBCELoss(nn.Module):
    """Binary cross-entropy loss that ignores missing (NaN) labels.

    Computes loss only on observed labels. NaN values in the target
    tensor are treated as unobserved and excluded from the loss.

    Args:
        pos_weight: Per-task positive class weights, shape (num_tasks,).
            If None, uses uniform weights.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else None,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked BCE loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape (batch, num_tasks).
            targets: Ground truth labels with NaN for missing,
                shape (batch, num_tasks).

        Returns:
            Scalar loss averaged over observed entries.
        """
        # Mask: True where label is observed (not NaN)
        mask = ~torch.isnan(targets)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Replace NaN with 0 so BCE doesn't produce NaN gradients
        targets_safe = targets.clone()
        targets_safe[~mask] = 0.0

        # Compute per-element BCE with logits
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets_safe,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Zero out loss for unobserved entries and average over observed
        masked_loss = loss * mask.float()
        return masked_loss.sum() / mask.sum()


def compute_pos_weights(
    labels: np.ndarray,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """Compute per-task positive class weights from training labels.

    For each task: pos_weight = n_negative / n_positive.
    Clamped to max_weight to prevent training instability.

    Args:
        labels: Label array, shape (n_samples, n_tasks). NaN = missing.
        max_weight: Maximum allowed pos_weight.

    Returns:
        Tensor of pos_weights, shape (num_tasks,).
    """
    n_tasks = labels.shape[1]
    weights = []

    for i in range(n_tasks):
        col = labels[:, i]
        mask = ~np.isnan(col)
        observed = col[mask]

        n_pos = (observed == 1).sum()
        n_neg = (observed == 0).sum()

        if n_pos > 0:
            w = min(n_neg / n_pos, max_weight)
        else:
            w = 1.0

        weights.append(w)
        logger.debug(f"Task {i}: pos_weight={w:.1f} ({n_pos} pos, {n_neg} neg)")

    return torch.tensor(weights, dtype=torch.float32)
