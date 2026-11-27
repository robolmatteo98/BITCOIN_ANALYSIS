"""Descrizione e interpretazione trasparente dei cluster della role detection."""

from __future__ import annotations

import torch


EPSILON = 1e-9
DIRECTION_THRESHOLD = 0.20


def _feature_index(data) -> dict[str, int]:
    return {name: index for index, name in enumerate(data.feature_names)}


def _value(means: torch.Tensor, indices: dict[str, int], name: str) -> float:
    index = indices.get(name)
    return float(means[index]) if index is not None else 0.0


def _cluster_profile(data, means: torch.Tensor) -> dict[str, float]:
    """Calcola indicatori interpretabili dalle medie trasformate del cluster."""
    indices = _feature_index(data)
    incoming = (
        _value(means, indices, "incoming_degree")
        + _value(means, indices, "incoming_counterparties")
        + _value(means, indices, "total_received")
    ) / 3.0
    outgoing = (
        _value(means, indices, "outgoing_degree")
        + _value(means, indices, "outgoing_counterparties")
        + _value(means, indices, "total_sent")
    ) / 3.0
    directionality = (incoming - outgoing) / (incoming + outgoing + EPSILON)
    balance = 1.0 - abs(directionality)
    structural_activity = (
        _value(means, indices, "incoming_degree")
        + _value(means, indices, "outgoing_degree")
        + _value(means, indices, "incoming_counterparties")
        + _value(means, indices, "outgoing_counterparties")
    ) / 4.0
    economic_activity = (
        _value(means, indices, "total_received")
        + _value(means, indices, "total_sent")
    ) / 2.0
    return {
        "incoming_pressure": incoming,
        "outgoing_pressure": outgoing,
        "directionality": directionality,
        "balance": balance,
        "structural_activity": structural_activity,
        "economic_activity": economic_activity,
        "hub_index": (structural_activity + economic_activity) / 2.0,
        "total_received": _value(means, indices, "total_received"),
        "total_sent": _value(means, indices, "total_sent"),
        "net_flow": _value(means, indices, "net_flow"),
        "active_days": _value(means, indices, "active_days"),
        "wallet_pagerank_basic": _value(means, indices, "wallet_pagerank_basic"),
        "address_pagerank_basic": _value(means, indices, "address_pagerank_basic"),
    }


def describe_v1_clusters(data, assignments: torch.Tensor) -> None:
    """Stampa le medie delle feature principali per ogni cluster."""
    for cluster_id in sorted(set(assignments.tolist())):
        mask = assignments == cluster_id
        profile = _cluster_profile(data, data.role_features[mask].mean(dim=0))
        print(f"\nCluster {cluster_id}")
        print(f"  indirizzi: {int(mask.sum())}")
        if cluster_id == -1:
            print("  tipo: coinbase-like")
        print(f"  pressione entrante: {profile['incoming_pressure']:.3f}")
        print(f"  pressione uscente: {profile['outgoing_pressure']:.3f}")
        print(f"  direzionalità [-1,1]: {profile['directionality']:.3f}")
        print(f"  equilibrio [0,1]: {profile['balance']:.3f}")
        print(f"  indice hub: {profile['hub_index']:.3f}")
        print(f"  totale ricevuto trasformato: {profile['total_received']:.3f}")
        print(f"  totale inviato trasformato: {profile['total_sent']:.3f}")
        print(f"  giorni attivi trasformati: {profile['active_days']:.3f}")


def assign_roles(data, assignments: torch.Tensor) -> dict[int, dict[str, object]]:
    """Assegna etichette descrittive con indicatori espliciti e verificabili."""
    cluster_ids = sorted(set(assignments.tolist()))
    profiles = {}
    members_by_cluster = {}
    for cluster_id in cluster_ids:
        members_by_cluster[cluster_id] = torch.where(assignments == cluster_id)[0]
        profiles[cluster_id] = _cluster_profile(
            data, data.role_features[members_by_cluster[cluster_id]].mean(dim=0)
        )
    hub_max = max((p["hub_index"] for p in profiles.values()), default=1.0)

    roles: dict[int, dict[str, object]] = {}
    for cluster_id in cluster_ids:
        member_indices = members_by_cluster[cluster_id]
        profile = profiles[cluster_id]
        if cluster_id == -1:
            representative_index = member_indices[0].item()
            role = "coinbase-like"
        else:
            directionality = profile["directionality"]
            relative_hub = profile["hub_index"] / (hub_max + EPSILON)
            if directionality >= DIRECTION_THRESHOLD:
                role = "collector-like"
            elif directionality <= -DIRECTION_THRESHOLD:
                role = "distributor-like"
            elif relative_hub >= 0.90:
                role = "hub-like"
            else:
                role = "user-like"
            activity_scores = data.role_features[member_indices, 0] + data.role_features[member_indices, 1]
            representative_index = member_indices[activity_scores.argmax()].item()

        roles[cluster_id] = {
            "role": role,
            "size": int(len(member_indices)),
            "representative": data.node_names[representative_index],
            "evidence": {
                **profile,
                "relative_hub_index": profile["hub_index"] / (hub_max + EPSILON),
                "direction_threshold": DIRECTION_THRESHOLD,
            },
        }
    return roles
