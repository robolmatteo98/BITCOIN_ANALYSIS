"""Disegno del grafo con i nodi colorati secondo il cluster K-Means."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D


def plot_clustered_graph(data, assignments: torch.Tensor, output: str | Path, title: str = "Grafo degli indirizzi clusterizzati", seed: int = 42) -> Path:
    """Salva il grafo degli archi osservati con colori per cluster."""
    if len(assignments) != len(data.node_names):
        raise ValueError("assignments deve avere un valore per ogni nodo")

    graph = nx.Graph()
    graph.add_nodes_from(range(len(data.node_names)))
    edges = data.positive_edges.detach().cpu().T.tolist()
    graph.add_edges_from((int(source), int(target)) for source, target in edges)

    positions = nx.spring_layout(graph, seed=seed)
    values = assignments.detach().cpu().tolist()
    cluster_ids = sorted(set(int(value) for value in values))
    palette = plt.get_cmap("tab10")
    colors = {
        cluster_id: ("black" if cluster_id == -1 else palette(index % 10))
        for index, cluster_id in enumerate(cluster_ids)
    }
    node_colors = [colors[int(value)] for value in values]

    figure_size = max(12.0, min(24.0, 8.0 + len(data.node_names) / 20.0))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    nx.draw_networkx_edges(graph, positions, ax=axis, edge_color="#999999", alpha=0.35, width=0.8, arrows=False)
    nx.draw_networkx_nodes(graph, positions, ax=axis, node_color=node_colors, node_size=350, edgecolors="white", linewidths=0.7)
    labels = {index: str(address) for index, address in enumerate(data.node_names)}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=6, bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.2})

    legend = []
    for cluster_id in cluster_ids:
        label = "coinbase-like" if cluster_id == -1 else f"cluster {cluster_id}"
        legend.append(Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=colors[cluster_id], markersize=9))
    axis.legend(handles=legend, loc="upper left", title="Cluster")
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_embedding_clusters(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    output: str | Path,
    title: str = "Cluster K-Means degli embedding",
) -> Path:
    """Salva una proiezione 2D degli embedding, senza mostrare gli archi."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings deve essere una matrice 2D")
    if len(embeddings) != len(assignments):
        raise ValueError("embeddings e assignments devono avere lo stesso numero di nodi")
    if len(embeddings) == 0:
        raise ValueError("non è possibile visualizzare embedding vuoti")

    values = assignments.detach().cpu().tolist()
    points = embeddings.detach().cpu().float()
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    if projected.shape[1] == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)

    cluster_ids = sorted(set(int(value) for value in values))
    palette = plt.get_cmap("tab10")
    colors = {
        cluster_id: ("black" if cluster_id == -1 else palette(index % 10))
        for index, cluster_id in enumerate(cluster_ids)
    }

    figure, axis = plt.subplots(figsize=(12, 9))
    for cluster_id in cluster_ids:
        mask = torch.tensor([value == cluster_id for value in values])
        label = (
            f"coinbase-like ({int(mask.sum())})"
            if cluster_id == -1
            else f"cluster {cluster_id} ({int(mask.sum())})"
        )
        axis.scatter(
            projected[mask, 0], projected[mask, 1], s=28, alpha=0.75,
            color=colors[cluster_id], label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("Componente principale 1")
    axis.set_ylabel("Componente principale 2")
    axis.grid(alpha=0.2)
    axis.legend(title="Gruppi", loc="best")
    figure.tight_layout()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
