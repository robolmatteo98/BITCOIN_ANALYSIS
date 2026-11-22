"""Costruzione comune del grafo e delle feature per la role detection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class GraphData:
    node_names: list[str]
    X: torch.Tensor
    P: torch.Tensor
    positive_edges: torch.Tensor
    negative_edges: torch.Tensor
    labels: torch.Tensor
    structural_features: torch.Tensor


def _sample_negative_edges(
    num_nodes: int,
    positive_edges: set[tuple[int, int]],
    count: int,
    seed: int,
) -> torch.Tensor:
    """Campiona negativi senza materializzare tutte le coppie possibili."""
    possible = num_nodes * (num_nodes - 1) - len(positive_edges)
    if possible < count:
        raise ValueError("non ci sono abbastanza coppie negative disponibili")

    generator = random.Random(seed)
    negative: set[tuple[int, int]] = set()
    while len(negative) < count:
        source = generator.randrange(num_nodes)
        target = generator.randrange(num_nodes)
        pair = (source, target)
        if source != target and pair not in positive_edges:
            negative.add(pair)

    return torch.tensor(sorted(negative), dtype=torch.long).T.contiguous()


def build_graph_data(
    transactions: Iterable[tuple[str, str, float, object]],
    node_names: Iterable[str] | None = None,
    seed: int = 42,
) -> GraphData:
    """Costruisce GraphData da tuple ``(source, destination, amount, time)``."""
    transactions = list(transactions)
    if not transactions:
        raise ValueError("nessuna transazione utilizzabile")

    # Creo la lista dei nodi. Nel dataset statico arriva da NODE_NAMES; nel
    # dataset DB viene ricavata dagli indirizzi source e destination.
    names = list(node_names) if node_names is not None else []
    for source, target, _, _ in transactions:
        if source not in names:
            names.append(source)
        if target not in names:
            names.append(target)

    names = [str(name) for name in names]
    node_to_id = {name: index for index, name in enumerate(names)}
    num_nodes = len(names)

    # Creo una lista di indici = nodo e una matrice di adiacenza inizialmente
    # tutta a zero.
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    # Vettori con caratteristiche strutturali per ogni nodo.
    in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    btc_in = torch.zeros(num_nodes, dtype=torch.float32)
    btc_out = torch.zeros(num_nodes, dtype=torch.float32)
    incoming_counterparties = [set() for _ in names]
    outgoing_counterparties = [set() for _ in names]
    positive_pairs: set[tuple[int, int]] = set()

    # Valorizzo adiacenza, gradi, importi e numero di controparti.
    for source_name, target_name, amount, _ in transactions:
        source = node_to_id[str(source_name)]
        target = node_to_id[str(target_name)]
        amount = float(amount)

        adjacency[source, target] += 1.0
        out_degree[source] += 1.0
        in_degree[target] += 1.0
        btc_out[source] += amount
        btc_in[target] += amount
        outgoing_counterparties[source].add(target)
        incoming_counterparties[target].add(source)
        positive_pairs.add((source, target))

    # Normalizzo con log1p le feature di grado e importo per evitare che i
    # valori molto grandi dominino il modello.
    structural_features = torch.stack(
        [
            torch.log1p(in_degree),
            torch.log1p(out_degree),
            torch.log1p(btc_in),
            torch.log1p(btc_out),
            torch.tensor(
                [len(values) for values in incoming_counterparties],
                dtype=torch.float32,
            ),
            torch.tensor(
                [len(values) for values in outgoing_counterparties],
                dtype=torch.float32,
            ),
        ],
        dim=1,
    )

    feature_max = structural_features.max(dim=0, keepdim=True).values
    feature_max = torch.where(
        feature_max > 0, feature_max, torch.ones_like(feature_max)
    )
    features = structural_features / feature_max

    # P[dest, source]: matrice usata dalla GCN per aggregare i vicini entranti.
    # La trasposta mantiene la stessa convenzione del dataset statico, poi
    # aggiungiamo l'identità per includere le feature del nodo stesso.
    propagation = adjacency.T + torch.eye(num_nodes)
    propagation = propagation / propagation.sum(dim=1, keepdim=True)

    # Gli archi positivi vengono deduplicati: più flussi tra la stessa coppia
    # di nodi rappresentano comunque un solo arco per il training del GAE.
    positive_edges = torch.tensor(
        sorted(positive_pairs), dtype=torch.long
    ).T.contiguous()

    # Genero tanti archi negativi quanti sono i positivi, evitando self-loop e
    # coppie già presenti nel grafo.
    negative_edges = _sample_negative_edges(
        num_nodes, positive_pairs, positive_edges.shape[1], seed
    )

    # Le label servono alla BCEWithLogitsLoss del graph autoencoder:
    # 1 per gli archi osservati, 0 per quelli campionati casualmente.
    labels = torch.cat(
        [
            torch.ones(positive_edges.shape[1]),
            torch.zeros(negative_edges.shape[1]),
        ]
    )

    return GraphData(
        node_names=names,
        X=features,
        P=propagation,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        labels=labels,
        structural_features=structural_features,
    )
