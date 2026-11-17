"""Strutture dati della role detection v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphData:
    """Grafo e feature pronti per il graph autoencoder."""

    node_names: list[str]
    feature_names: list[str]
    X: torch.Tensor
    P: torch.Tensor
    positive_edges: torch.Tensor
    negative_edges: torch.Tensor
    labels: torch.Tensor
    # Feature trasformate ma non standardizzate, usate per interpretare i ruoli.
    role_features: torch.Tensor
    # Maschera dei nodi che hanno ricevuto almeno un output coinbase.
    coinbase_nodes: torch.Tensor
