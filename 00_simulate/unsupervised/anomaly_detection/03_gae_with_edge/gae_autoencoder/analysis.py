"""Valutazione degli archi e degli anomaly score."""

import torch

from .data import GraphData
from .models import GraphAutoencoder


def calculate_anomaly_edge(probability: torch.Tensor) -> float:
    """Calcola l'anomaly score di un arco osservato."""
    return -torch.log(probability.clamp_min(1e-8)).item()


def edge_probabilities(model, data):
    model.eval()
    with torch.no_grad():
        _, logits = model(data.X, data.P, data.edge_features)
        return torch.sigmoid(logits)


def print_split_results(model: GraphAutoencoder, data: GraphData, split_id: int) -> None:
    probabilities = edge_probabilities(model, data)
    print(f"\nSplit {split_id}")
    for src, dst in data.test_edges:
        print(
            f"  {data.node_names[src]} -> {data.node_names[dst]} | "
            f"Probabilità: {probabilities[src, dst].item():.6f}"
        )


def evaluate_hidden_edges(model: GraphAutoencoder, data: GraphData):
    probabilities = edge_probabilities(model, data)
    node_errors = {name: [] for name in data.node_names}
    print("\n" + "=" * 55)
    print("RICOSTRUZIONE DEGLI ARCHI NASCOSTI")
    print("=" * 55)
    for src, dst in data.test_edges:
        probability = probabilities[src, dst].item()
        error = calculate_anomaly_edge(probabilities[src, dst])
        src_name, dst_name = data.node_names[src], data.node_names[dst]
        print(f"{src_name} -> {dst_name} | Probabilità: {probability:.6f} | Errore: {error:.6f}")
        node_errors[src_name].append(error)
        node_errors[dst_name].append(error)
    return node_errors, None


def print_edge_probability_comparison(model: GraphAutoencoder, data: GraphData) -> None:
    probabilities = edge_probabilities(model, data)
    print("\n" + "=" * 55)
    print("CONFRONTO PROBABILITÀ ARCHI")
    print("=" * 55)
    for title, edges in (
        ("Positivi usati nel training:", data.train_edges),
        ("Negativi usati nel training:", data.train_negative_edges),
        ("Positivi nascosti usati come test:", data.test_edges),
    ):
        print("\n" + title)
        for src, dst in edges:
            print(
                f"  {data.node_names[src]} -> {data.node_names[dst]} | "
                f"Probabilità: {probabilities[src, dst].item():.6f}"
            )


def calculate_anomaly_scores(node_errors):
    return {
        name: (sum(errors) / len(errors) if errors else None)
        for name, errors in node_errors.items()
    }


def print_anomaly_scores(scores) -> None:
    print("\n" + "=" * 55)
    print("ANOMALY SCORE DEI WALLET")
    print("=" * 55)
    ranked = sorted(
        ((name, score) for name, score in scores.items() if score is not None),
        key=lambda item: item[1], reverse=True,
    )
    for name, score in ranked:
        print(f"Wallet {name} | Anomaly score: {score:.6f}")
    for name, score in scores.items():
        if score is None:
            print(f"Wallet {name} | Nessun arco di test incidente: score non disponibile")
