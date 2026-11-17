"""Configurazione e dati di esempio del graph autoencoder."""

import random

import torch

NODE_NAMES = ["A", "B", "C", "D", "E", "F", "G"]
TRANSACTIONS = [
    ("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"),
    ("B", "C"), ("C", "D"), ("C", "E"), ("D", "B"),
    ("E", "F"), ("E", "G"), ("F", "G"), ("G", "D"),
]
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
