"""Implementazione PyTorch di K-Means per gli embedding dei nodi."""

from __future__ import annotations

import torch


def kmeans_assignments(embeddings: torch.Tensor, n_clusters: int, iterations: int = 100) -> torch.Tensor:
    """Assegna ogni embedding a un cluster K-Means."""
    assignments, _ = kmeans_fit(embeddings, n_clusters=n_clusters, iterations=iterations)
    return assignments


def kmeans_fit(embeddings: torch.Tensor, n_clusters: int, iterations: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Apprende centroidi e assegnamenti K-Means sugli embedding forniti."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings deve essere una matrice 2D")
    if n_clusters < 1 or n_clusters > embeddings.shape[0]:
        raise ValueError("n_clusters deve essere compreso tra 1 e il numero di nodi")
    if iterations < 1:
        raise ValueError("iterations deve essere positivo")

    # vengono scelti i centroidi iniziali per K-Means
    generator = torch.Generator(device=embeddings.device).manual_seed(42)

    # embedding.shaep[0] è il numero di nodi, viene creata una permutazione casuale degli indici
    initial_indices = torch.randperm(embeddings.shape[0], generator=generator)[:n_clusters]

    # seleziona gli embedding da quei nodi e li usa come centroidi iniziali
    centroids = embeddings[initial_indices].clone()

    for _ in range(iterations):

        # calcola la distanza tra ogni embedding e ogni centroide
        distances = torch.cdist(embeddings, centroids)

        # assegna ogni nodo al centroide più vicino
        assignments = distances.argmin(dim=1)
        updated_centroids = []

        # vengono ricalcolati i centroidi
        for cluster_id in range(n_clusters):

            # seleziona gli embedding assegnati a quei cluster
            members = embeddings[assignments == cluster_id]

            # calcola il punto medio dei membri del cluster. Questo diventa il nuovo centroide.
            # se il cluster è vuoto mantiene il vecchio centroide
            updated_centroids.append(
                centroids[cluster_id] if len(members) == 0 else members.mean(dim=0)
            )

        # ricostruisce la matrice dei centroidi
        new_centroids = torch.stack(updated_centroids)

        # se i centroidi nuovi sono praticamente uguali ai precedenti, l’algoritmo è convergente e il ciclo termina
        if torch.allclose(new_centroids, centroids):
            break

        # altrimenti, usa i nuovi centroidi per l’iterazione successiva
        centroids = new_centroids

    return assignments, centroids


def kmeans_predict(embeddings: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Assegna nuovi embedding ai centroidi già appresi."""
    if embeddings.ndim != 2 or centroids.ndim != 2:
        raise ValueError("embeddings e centroids devono essere matrici 2D")
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError("embedding e centroidi devono avere la stessa dimensione")
    if centroids.shape[0] < 1:
        raise ValueError("serve almeno un centroide")
    return torch.cdist(embeddings, centroids).argmin(dim=1)
