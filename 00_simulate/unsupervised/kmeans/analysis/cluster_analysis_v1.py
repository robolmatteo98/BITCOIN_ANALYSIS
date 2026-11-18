"""Analisi descrittiva dei cluster della role detection v1."""

from __future__ import annotations

import torch


def describe_v1_clusters(data, assignments: torch.Tensor) -> None:
    """Stampa le medie delle feature principali per ogni cluster."""
    for cluster_id in sorted(set(assignments.tolist())):
        mask = assignments == cluster_id
        mean_values = data.role_features[mask].mean(dim=0)
        print(f"\nCluster {cluster_id}")
        print(f"  indirizzi: {int(mask.sum())}")
        if cluster_id == -1:
            print("  tipo: coinbase-like")
        print(f"  ingressi: {mean_values[0]:.2f}")
        print(f"  uscite: {mean_values[1]:.2f}")
        print(f"  controparti entranti: {mean_values[2]:.2f}")
        print(f"  controparti uscenti: {mean_values[3]:.2f}")
        print(f"  totale ricevuto: {mean_values[9]:.2f}")
        print(f"  totale inviato: {mean_values[10]:.2f}")
        print(f"  giorni attivi: {mean_values[16]:.2f}")


def assign_roles(data, assignments: torch.Tensor) -> dict[int, dict[str, object]]:
    """Assegna ruoli euristici e restituisce un rappresentante per cluster."""
    cluster_ids = sorted(set(assignments.tolist()))
    cluster_means = torch.stack([
        data.role_features[assignments == cluster_id].mean(dim=0)
        for cluster_id in cluster_ids
    ])
    feature_max = cluster_means.max(dim=0, keepdim=True).values
    feature_max = torch.where(feature_max > 0, feature_max, torch.ones_like(feature_max))
    normalized = cluster_means / feature_max

    roles: dict[int, dict[str, object]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        member_indices = torch.where(assignments == cluster_id)[0]
        if cluster_id == -1:
            representative_index = member_indices[0].item()
            role = "coinbase-like"
        else:
            incoming = normalized[index, 0] + normalized[index, 2] + normalized[index, 9]
            outgoing = normalized[index, 1] + normalized[index, 3] + normalized[index, 10]
            activity = normalized[index].sum()
            if incoming > outgoing * 1.15:
                role = "collector-like"
            elif outgoing > incoming * 1.15:
                role = "distributor-like"
            elif activity == normalized.sum(dim=1).max():
                role = "hub-like"
            else:
                role = "user-like"
            activity_scores = data.role_features[member_indices, 0] + data.role_features[member_indices, 1]
            representative_index = member_indices[activity_scores.argmax()].item()

        roles[cluster_id] = {
            "role": role,
            "size": int(len(member_indices)),
            "representative": data.node_names[representative_index],
        }
    return roles
