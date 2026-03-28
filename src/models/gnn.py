"""GNN models: GCN and GIN for molecular property prediction."""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from src.models.multitask import MultiTaskHead

logger = logging.getLogger(__name__)

POOLING_MAP = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
}


class GCNModel(nn.Module):
    """Graph Convolutional Network for multi-task molecular prediction.

    Architecture: N × (GCNConv → BatchNorm → ReLU → Dropout) → Pooling → Head.

    Args:
        in_dim: Input node feature dimensionality (140 for Tox21).
        hidden_dim: Hidden layer dimensionality.
        num_layers: Number of GCN message-passing layers.
        dropout: Dropout rate between layers.
        pooling: Global pooling strategy ('mean', 'add', 'max').
        head_hidden_dim: Hidden dim of the multi-task prediction head.
        num_tasks: Number of output tasks.
        head_dropout: Dropout in the prediction head.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean",
        head_hidden_dim: int = 128,
        num_tasks: int = 12,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer: in_dim → hidden_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool_fn = POOLING_MAP[pooling]
        self.head = MultiTaskHead(
            in_dim=hidden_dim,
            hidden_dim=head_hidden_dim,
            num_tasks=num_tasks,
            dropout=head_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features, shape (total_nodes, in_dim).
            edge_index: Edge index, shape (2, total_edges).
            batch: Batch assignment vector, shape (total_nodes,).

        Returns:
            Logits, shape (batch_size, num_tasks).
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        # Global pooling: (total_nodes, hidden) → (batch_size, hidden)
        x = self.pool_fn(x, batch)

        return self.head(x)


class GINModel(nn.Module):
    """Graph Isomorphism Network for multi-task molecular prediction.

    Architecture: N × (GINConv(MLP) → BatchNorm → ReLU → Dropout) → Pooling → Head.
    Each GINConv layer uses a 2-layer MLP as its update function.

    Args:
        in_dim: Input node feature dimensionality.
        hidden_dim: Hidden layer dimensionality.
        num_layers: Number of GIN message-passing layers.
        dropout: Dropout rate between layers.
        pooling: Global pooling strategy ('mean', 'add', 'max').
        train_eps: Whether to learn the epsilon parameter in GIN.
        head_hidden_dim: Hidden dim of the multi-task prediction head.
        num_tasks: Number of output tasks.
        head_dropout: Dropout in the prediction head.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "add",
        train_eps: bool = True,
        head_hidden_dim: int = 128,
        num_tasks: int = 12,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # First layer: in_dim → hidden_dim
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(nn=mlp, train_eps=train_eps))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.pool_fn = POOLING_MAP[pooling]
        self.head = MultiTaskHead(
            in_dim=hidden_dim,
            hidden_dim=head_hidden_dim,
            num_tasks=num_tasks,
            dropout=head_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features, shape (total_nodes, in_dim).
            edge_index: Edge index, shape (2, total_edges).
            batch: Batch assignment vector, shape (total_nodes,).

        Returns:
            Logits, shape (batch_size, num_tasks).
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        x = self.pool_fn(x, batch)

        return self.head(x)


def build_model(config: dict, in_dim: int) -> nn.Module:
    """Build a GNN model from a configuration dictionary.

    Args:
        config: Model configuration from YAML.
        in_dim: Input node feature dimensionality.

    Returns:
        GNN model instance.
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    head_cfg = model_cfg.get("head", {})

    common_kwargs = dict(
        in_dim=in_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.2),
        pooling=model_cfg.get("pooling", "mean"),
        head_hidden_dim=head_cfg.get("hidden_dim", 128),
        num_tasks=head_cfg.get("num_tasks", 12),
        head_dropout=head_cfg.get("dropout", 0.1),
    )

    if model_name == "gcn":
        model = GCNModel(**common_kwargs)
    elif model_name == "gin":
        gin_cfg = model_cfg.get("gin", {})
        model = GINModel(
            **common_kwargs,
            train_eps=gin_cfg.get("train_eps", True),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built {model_name.upper()} with {n_params:,} parameters")

    return model
