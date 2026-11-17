"""Analisi degli archi e degli anomaly score."""

import torch


def calculate_anomaly_edge(probability: torch.Tensor) -> float:
    return -torch.log(probability.clamp_min(1e-8)).item()
