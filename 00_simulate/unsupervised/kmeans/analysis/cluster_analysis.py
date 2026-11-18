"""Analisi descrittiva dei cluster della role detection originale."""

import torch


FEATURE_NAMES = [
    "grado entrante",
    "grado uscente",
    "BTC entrati",
    "BTC usciti",
    "controparti entranti",
    "controparti uscenti",
]


def describe_clusters(data, assignments):
    """Stampa le statistiche medie delle feature per ogni cluster."""
    features = data.structural_features
    summaries = {}
    for cluster_id in sorted(set(assignments.tolist())):
        mask = assignments == cluster_id
        mean_values = features[mask].mean(dim=0)
        summaries[cluster_id] = mean_values
        print(f"\nCluster {cluster_id}")
        print("  " + " | ".join(
            f"{name}: {value:.2f}"
            for name, value in zip(FEATURE_NAMES, mean_values)
        ))
    return summaries


def interpret_clusters(data, assignments):
    """Assegna un ruolo debole ai cluster usando le feature strutturali."""
    features = data.structural_features
    cluster_ids = sorted(set(assignments.tolist()))
    cluster_means = torch.stack([
        features[assignments == cluster_id].mean(dim=0)
        for cluster_id in cluster_ids
    ])

    feature_max = cluster_means.max(dim=0, keepdim=True).values
    feature_max = torch.where(feature_max > 0, feature_max, torch.ones_like(feature_max))
    normalized = cluster_means / feature_max

    roles = {}
    for index, cluster_id in enumerate(cluster_ids):
        incoming = normalized[index, 0] + normalized[index, 2] + normalized[index, 4]
        outgoing = normalized[index, 1] + normalized[index, 3] + normalized[index, 5]
        activity = normalized[index].sum()

        if incoming > outgoing * 1.15:
            role = "collector-like"
        elif outgoing > incoming * 1.15:
            role = "distributor-like"
        elif activity == normalized.sum(dim=1).max():
            role = "hub-like"
        else:
            role = "user-like"
        roles[cluster_id] = role

    print("\nInterpretazione debole dei cluster:")
    for cluster_id in cluster_ids:
        member_indices = torch.where(assignments == cluster_id)[0]
        members = [data.node_names[index] for index in member_indices.tolist()]
        cluster_features = features[member_indices]
        representative = data.node_names[
            member_indices[(cluster_features[:, 0] + cluster_features[:, 1]).argmax()].item()
        ]
        print(f"  Cluster {cluster_id}: {roles[cluster_id]}")
        print(f"    Wallet: {', '.join(members)}")
        print(f"    Wallet più rappresentativo: {representative}")
    return roles
