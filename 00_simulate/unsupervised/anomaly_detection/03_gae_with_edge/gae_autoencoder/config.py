"""Configurazione e feature degli archi del grafo didattico."""

import random

import torch

NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G"]
TRANSACTIONS = [
    ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"),
    ("B", "C"), ("C", "D"), ("C", "E"), ("D", "B"),
    ("E", "F"), ("E", "G"), ("F", "G"), ("G", "D"),
]

# Feature edge didattiche: [importo, timestamp].
EDGE_FEATURES = {
    ("A", "B"): [10.0, 1.0], ("A", "C"): [4.0, 2.0],
    ("A", "D"): [8.0, 3.0], ("A", "E"): [2.0, 4.0],
    ("B", "C"): [7.0, 5.0], ("C", "D"): [3.0, 6.0],
    ("C", "E"): [6.0, 7.0], ("D", "B"): [5.0, 8.0],
    ("E", "F"): [9.0, 9.0], ("E", "G"): [1.0, 10.0],
    ("F", "G"): [11.0, 11.0], ("G", "D"): [12.0, 12.0],
}

TEST_SPLITS = [
    [("A", "B"), ("C", "D"), ("E", "G")],
    [("A", "C"), ("C", "E"), ("F", "G")],
    [("A", "D"), ("B", "C"), ("G", "D")],
]
HIDDEN_DIM = 8
LATENT_DIM = 4
EDGE_DIM = 2
LEARNING_RATE = 0.01
EPOCHS = 500


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
