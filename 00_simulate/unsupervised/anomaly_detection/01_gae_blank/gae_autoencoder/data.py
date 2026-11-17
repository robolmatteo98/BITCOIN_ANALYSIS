"""Preparazione di nodi, archi, matrici e campioni di training."""

from dataclasses import dataclass
import random
from typing import List, Tuple

import torch

from .config import NODE_NAMES, TRANSACTIONS


@dataclass
class GraphData:
    node_names: list[str]
    all_edges: list[tuple[int, int]]
    train_edges: list[tuple[int, int]]
    train_negative_edges: list[tuple[int, int]]
    test_edges: list[tuple[int, int]]
    X: torch.Tensor
    P: torch.Tensor
    train_pairs: torch.Tensor
    train_labels: torch.Tensor

    @property
    def node_to_id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.node_names)}


def prepare_graph(
    test_transactions: List[Tuple[str, str]],
    seed: int = 42,
) -> GraphData:
    node_to_id = {name: i for i, name in enumerate(NODE_NAMES)}
    all_edges = [(node_to_id[src], node_to_id[dst]) for src, dst in TRANSACTIONS]

    # trasforma le transazioni (wallet, wallet) a (id_wallet, id_wallet)
    test_edges = [(node_to_id[src], node_to_id[dst]) for src, dst in test_transactions]
    test_set = set(test_edges)
    train_edges = [edge for edge in all_edges if edge not in test_set]
    num_nodes = len(NODE_NAMES)

    # CREO la matrice di adiacenza, assegnando 1.0 dove compare la relazione (solo training)
    A_train = torch.zeros(num_nodes, num_nodes)
    for src, dst in train_edges:
        A_train[src, dst] = 1.0

    # Trasposta della matrice + diagonale di ogni nodo (matrice identità)
    B = A_train.T + torch.eye(num_nodes)

    # Normalizzo ogni riga
    P = B / B.sum(dim=1, keepdim=True)

    # Baseline: ogni nodo è rappresentato dalla propria identità one-hot.
    X = torch.eye(num_nodes)

    # Creo un insieme di coppie senza i self loop.
    all_pairs = [
        (i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    ]
    negative_edges = [edge for edge in all_pairs if edge not in set(all_edges)]
    random.Random(seed).shuffle(negative_edges)
    train_negative = negative_edges[:len(train_edges)]

    return GraphData(
        node_names=NODE_NAMES,
        all_edges=all_edges,
        train_edges=train_edges,
        train_negative_edges=train_negative,
        test_edges=test_edges,
        X=X,
        P=P,
        train_pairs=torch.tensor(train_edges + train_negative, dtype=torch.long),
        train_labels=torch.tensor(
            [1.0] * len(train_edges) + [0.0] * len(train_negative),
            dtype=torch.float32,
        ),
    )
