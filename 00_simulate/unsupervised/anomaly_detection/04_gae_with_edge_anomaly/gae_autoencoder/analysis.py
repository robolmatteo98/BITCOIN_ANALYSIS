"""Anomaly score per transazione osservata nel futuro."""
import torch

def calculate_anomaly_edge(probability: torch.Tensor) -> float:
    return -torch.log(probability.clamp_min(1e-8)).item()


def classify_anomaly(probability: float) -> str:
    """Converte la probabilità stimata in una classe leggibile."""
    if probability <= 0.10:
        return "anomalia probabile"
    if probability >= 0.90:
        return "anomalia assente"
    return "indeciso"

def score_observed_transactions(model, data):
    model.eval()
    with torch.no_grad():
        Z = model.encode(data.X, data.train_edge_index, data.train_edge_features)
        logits = model.score(Z, data.observed_edge_index, data.observed_edge_features)
        probabilities = torch.sigmoid(logits)
    return [
        (*tx, p.item(), calculate_anomaly_edge(p), classify_anomaly(p.item()))
        for tx, p in zip(data.observed_transactions, probabilities)
    ]
