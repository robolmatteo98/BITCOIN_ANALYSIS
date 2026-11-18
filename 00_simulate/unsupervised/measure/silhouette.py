"""Calcolo della silhouette score con PyTorch."""

from __future__ import annotations

import torch


def silhouette_score(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    sample_size: int | None = None,
    seed: int = 42,
) -> float:
    """Calcola la silhouette media degli embedding assegnati ai cluster.

    La misura varia tra -1 e 1:
    - valori vicini a 1 indicano cluster compatti e separati;
    - valori vicini a 0 indicano sovrapposizione;
    - valori negativi indicano possibili assegnazioni problematiche.

    sample_size permette di limitare il costo quadratico su dataset grandi.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings deve essere una matrice 2D")
    if assignments.ndim != 1 or len(assignments) != len(embeddings):
        raise ValueError("assignments deve avere un valore per ogni embedding")
    if len(embeddings) < 2:
        return 0.0

    # labels: contiene il cluster assegnato ad ogni nodo
    # points: contiene gli embedding dei nodi, vettori prodotti da GCN e GAT
    labels = assignments.detach().cpu().long()
    points = embeddings.detach().cpu().float()

    # estrae gli identificativi dei cluster presenti
    cluster_ids = torch.unique(labels, sorted=True)
    if len(cluster_ids) < 2:
        return 0.0

    if sample_size is not None:
        if sample_size < 2:
            raise ValueError("sample_size deve essere almeno 2")
        if sample_size < len(points):
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(
                len(points), generator=generator
            )[:sample_size]
            points = points[indices]
            labels = labels[indices]
            cluster_ids = torch.unique(labels, sorted=True)
            if len(cluster_ids) < 2:
                return 0.0

    distances = torch.cdist(points, points)
    scores = []
    for index in range(len(points)):
        own_cluster = labels[index]
        same_cluster = labels == own_cluster
        same_cluster[index] = False

        if same_cluster.any():
            within_distance = distances[index, same_cluster].mean()
        else:
            within_distance = torch.tensor(0.0)

        between_distances = []
        for cluster_id in cluster_ids:
            if cluster_id == own_cluster:
                continue
            other_cluster = labels == cluster_id
            if other_cluster.any():
                between_distances.append(distances[index, other_cluster].mean())

        if not between_distances:
            scores.append(torch.tensor(0.0))
            continue
        nearest_other_distance = torch.stack(between_distances).min()
        denominator = torch.maximum(within_distance, nearest_other_distance)
        if denominator == 0:
            scores.append(torch.tensor(0.0))
        else:
            scores.append(
                (nearest_other_distance - within_distance) / denominator
            )

    return float(torch.stack(scores).mean())
