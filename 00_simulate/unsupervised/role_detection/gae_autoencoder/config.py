"""Database sintetico statico condiviso con l'esperimento anomaly detection."""

import random
import torch

NODE_NAMES = list("ABCDEFGHIJKLMNOPQ")
TRANSACTIONS = [
    ("A", "B", 10.0, 1), ("A", "B", 3.0, 1), ("C", "D", 8.0, 1), ("E", "F", 5.0, 1), ("H", "I", 2.0, 1),
    ("A", "C", 4.0, 2), ("B", "D", 6.0, 2), ("G", "H", 7.0, 2), ("J", "K", 1.0, 2), ("L", "M", 9.0, 2),
    ("A", "B", 2.0, 3), ("D", "E", 11.0, 3), ("F", "G", 4.0, 3), ("N", "O", 6.0, 3), ("P", "Q", 12.0, 3), ("C", "D", 5.0, 3),
    ("A", "D", 8.0, 4), ("B", "C", 3.0, 4), ("E", "G", 7.0, 4), ("H", "J", 2.0, 4), ("K", "L", 10.0, 4),
    ("A", "B", 50.0, 5), ("C", "E", 1.0, 5), ("D", "F", 9.0, 5), ("G", "I", 3.0, 5), ("M", "N", 4.0, 5), ("O", "P", 8.0, 5),
    ("B", "E", 2.0, 6), ("A", "Q", 15.0, 6), ("C", "D", 6.0, 6), ("F", "H", 5.0, 6), ("J", "M", 7.0, 6),
    ("A", "B", 100.0, 7), ("E", "F", 1.0, 7), ("G", "D", 25.0, 7), ("K", "Q", 13.0, 7), ("H", "I", 2.0, 7),
    ("B", "A", 1.0, 8), ("C", "G", 17.0, 8), ("L", "P", 3.0, 8), ("N", "Q", 20.0, 8), ("D", "H", 6.0, 8),
    # Esempio di wallet hub/collector: 20 transazioni verso Q nello stesso blocco.
    ("A", "Q", 5.0, 9), ("B", "Q", 5.0, 9), ("C", "Q", 5.0, 9), ("D", "Q", 5.0, 9),
    ("E", "Q", 5.0, 9), ("F", "Q", 5.0, 9), ("G", "Q", 5.0, 9), ("H", "Q", 5.0, 9),
    ("I", "Q", 5.0, 9), ("J", "Q", 5.0, 9), ("K", "Q", 5.0, 9), ("L", "Q", 5.0, 9),
    ("M", "Q", 5.0, 9), ("N", "Q", 5.0, 9), ("O", "Q", 5.0, 9), ("P", "Q", 5.0, 9),
    ("A", "Q", 5.0, 9), ("B", "Q", 5.0, 9), ("C", "Q", 5.0, 9), ("D", "Q", 5.0, 9),
]
HIDDEN_DIM = 16
LATENT_DIM = 8
LEARNING_RATE = 0.01
EPOCHS = 500


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
