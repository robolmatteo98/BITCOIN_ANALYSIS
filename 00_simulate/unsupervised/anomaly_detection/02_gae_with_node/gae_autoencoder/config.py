"""Configurazione e dati di esempio del graph autoencoder."""

import random

import torch

NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G"]
TRANSACTIONS = [
    ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"),
    ("B", "C"), ("C", "D"), ("C", "E"), ("D", "B"),
    ("E", "F"), ("E", "G"), ("F", "G"), ("G", "D"),
]

# Importi didattici usati per costruire le feature aggregate dei nodi.
TRANSACTION_AMOUNTS = {
    ("A", "B"): 10.0, ("A", "C"): 4.0, ("A", "D"): 8.0,
    ("A", "E"): 2.0, ("B", "C"): 7.0, ("C", "D"): 3.0,
    ("C", "E"): 6.0, ("D", "B"): 5.0, ("E", "F"): 9.0,
    ("E", "G"): 1.0, ("F", "G"): 11.0, ("G", "D"): 12.0,
}
TEST_SPLITS = [
    [("A", "B"), ("C", "D"), ("E", "G")],
    [("A", "C"), ("C", "E"), ("F", "G")],
    [("A", "D"), ("B", "C"), ("G", "D")],
]
HIDDEN_DIM = 8
LATENT_DIM = 4
LEARNING_RATE = 0.01
EPOCHS = 500


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
