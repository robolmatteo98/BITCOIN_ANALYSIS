"""Dataset statico e adattatore verso il costruttore comune del grafo."""

from __future__ import annotations

from .config import NODE_NAMES, TRANSACTIONS
from .graph_builder import GraphData, build_graph_data


def prepare_graph(seed: int = 42) -> GraphData:
    """Costruisce il GraphData del dataset statico originale."""
    return build_graph_data(TRANSACTIONS, node_names=NODE_NAMES, seed=seed)
