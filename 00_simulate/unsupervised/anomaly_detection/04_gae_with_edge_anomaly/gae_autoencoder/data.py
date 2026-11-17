"""Preparazione di un grafo temporale con transazioni multi-edge."""
from dataclasses import dataclass
import random
import torch
from .config import EDGE_DIM, NODE_NAMES, TIME_CUTOFF, TRANSACTIONS

@dataclass
class GraphData:
    node_names: list[str]
    train_transactions: list[tuple[str, str, float, int]]
    observed_transactions: list[tuple[str, str, float, int]]
    train_edge_index: torch.Tensor
    train_edge_features: torch.Tensor
    observed_edge_index: torch.Tensor
    observed_edge_features: torch.Tensor
    X: torch.Tensor
    negative_edge_index: torch.Tensor

def _edge_tensors(transactions, node_to_id):
    edge_index = torch.tensor([[node_to_id[s], node_to_id[d]] for s, d, _, _ in transactions], dtype=torch.long).T
    edge_features = torch.tensor([[amount, timestamp] for _, _, amount, timestamp in transactions], dtype=torch.float32)
    return edge_index, edge_features

def prepare_graph(time_cutoff: int = TIME_CUTOFF, seed: int = 42) -> GraphData:

    # creo una lista di indici associati ai nomi dall'ordine di NODE_NAMES
    node_to_id = {name: i for i, name in enumerate(NODE_NAMES)}

    # seleziono le transazioni prima o uguale alla soglia per il training
    train_transactions = [tx for tx in TRANSACTIONS if tx[3] <= time_cutoff]

    # seleziono le altre transazioni per valutarle
    observed_transactions = [tx for tx in TRANSACTIONS if tx[3] > time_cutoff]

    # divido id sorgente e destinazione dalle caratteristiche della transazione (sia in train che in test)
    train_edge_index, train_edge_features = _edge_tensors(train_transactions, node_to_id)
    observed_edge_index, observed_edge_features = _edge_tensors(observed_transactions, node_to_id)

    # creo la matrice identità = N x N
    # creo la matrice delle feature, con 4 colonne per ogni nodo = N x 4
    node_identity = torch.eye(len(NODE_NAMES))
    node_features = torch.zeros(len(NODE_NAMES), 4)

    for src_name, dst_name, amount, _ in train_transactions:

        # converte i nomi dei nodi negli indici
        src, dst = node_to_id[src_name], node_to_id[dst_name]

        # aumenta il grado entrante, il grado uscente, BTCin e BTCout
        node_features[dst, 0] += 1.0
        node_features[src, 1] += 1.0
        node_features[dst, 2] += amount
        node_features[src, 3] += amount

    # normalizza i due valori
    amount_columns = torch.log1p(node_features[:, 2:4])

    # normalizza tra 0 e 1
    amount_max = amount_columns.max(dim=0, keepdim=True).values
    amount_max = torch.where(amount_max > 0, amount_max, torch.ones_like(amount_max))
    node_features[:, 2:4] = amount_columns / amount_max

    # concatena matrice identità con caratteristiche nodo = N x (N + 4)
    X = torch.cat([node_identity, node_features], dim=1)

    # Negativi strutturali: non sono etichette di anomalie.
    train_pairs = {(node_to_id[s], node_to_id[d]) for s, d, _, _ in train_transactions}
    all_pairs = [(i, j) for i in range(len(NODE_NAMES)) for j in range(len(NODE_NAMES)) if i != j]
    negative_pairs = [pair for pair in all_pairs if pair not in train_pairs]
    random.Random(seed).shuffle(negative_pairs)
    negative_edge_index = torch.tensor(negative_pairs[:len(train_transactions)], dtype=torch.long).T
    return GraphData(NODE_NAMES, train_transactions, observed_transactions, train_edge_index,
                     train_edge_features, observed_edge_index, observed_edge_features, X,
                     negative_edge_index)


def _aggregate_transactions(transactions):
    """Raggruppa le transazioni per coppia mantenendo le feature aggregate."""
    aggregated = {}
    for src, dst, amount, timestamp in transactions:
        key = (src, dst)
        if key not in aggregated:
            aggregated[key] = {
                "amount_total": 0.0,
                "timestamp_min": timestamp,
                "timestamp_max": timestamp,
                "transaction_count": 0,
            }
        values = aggregated[key]
        values["amount_total"] += amount
        values["timestamp_min"] = min(values["timestamp_min"], timestamp)
        values["timestamp_max"] = max(values["timestamp_max"], timestamp)
        values["transaction_count"] += 1

    # Manteniamo l'interfaccia delle transazioni a quattro elementi:
    # l'importo diventa il totale e il timestamp visualizzato diventa l'ultimo.
    return [
        (src, dst, values["amount_total"], values["timestamp_max"])
        for (src, dst), values in aggregated.items()
    ], aggregated


def _aggregated_edge_tensors(transactions, aggregated, node_to_id):
    edge_index = torch.tensor(
        [[node_to_id[src], node_to_id[dst]] for src, dst, _, _ in transactions],
        dtype=torch.long,
    ).T
    edge_features = torch.tensor(
        [
            [
                aggregated[(src, dst)]["amount_total"],
                aggregated[(src, dst)]["timestamp_min"],
                aggregated[(src, dst)]["timestamp_max"],
                aggregated[(src, dst)]["transaction_count"],
            ]
            for src, dst, _, _ in transactions
        ],
        dtype=torch.float32,
    )
    return edge_index, edge_features


def prepare_graph_aggregated(time_cutoff: int = TIME_CUTOFF, seed: int = 42) -> GraphData:
    """Prepara un grafo con un solo arco per coppia e feature aggregate."""
    node_to_id = {name: i for i, name in enumerate(NODE_NAMES)}
    train_raw = [tx for tx in TRANSACTIONS if tx[3] <= time_cutoff]
    observed_raw = [tx for tx in TRANSACTIONS if tx[3] > time_cutoff]
    train_transactions, train_aggregated = _aggregate_transactions(train_raw)
    observed_transactions, observed_aggregated = _aggregate_transactions(observed_raw)
    train_edge_index, train_edge_features = _aggregated_edge_tensors(
        train_transactions, train_aggregated, node_to_id
    )
    observed_edge_index, observed_edge_features = _aggregated_edge_tensors(
        observed_transactions, observed_aggregated, node_to_id
    )

    # Le feature dei nodi continuano a essere calcolate sulle transazioni
    # storiche originali, prima dell'aggregazione degli archi.
    node_identity = torch.eye(len(NODE_NAMES))
    node_features = torch.zeros(len(NODE_NAMES), 4)
    for src_name, dst_name, amount, _ in train_raw:
        src, dst = node_to_id[src_name], node_to_id[dst_name]
        node_features[dst, 0] += 1.0
        node_features[src, 1] += 1.0
        node_features[dst, 2] += amount
        node_features[src, 3] += amount
    amount_columns = torch.log1p(node_features[:, 2:4])
    amount_max = amount_columns.max(dim=0, keepdim=True).values
    amount_max = torch.where(amount_max > 0, amount_max, torch.ones_like(amount_max))
    node_features[:, 2:4] = amount_columns / amount_max
    X = torch.cat([node_identity, node_features], dim=1)

    train_pairs = {(node_to_id[src], node_to_id[dst]) for src, dst, _, _ in train_transactions}
    all_pairs = [(i, j) for i in range(len(NODE_NAMES)) for j in range(len(NODE_NAMES)) if i != j]
    negative_pairs = [pair for pair in all_pairs if pair not in train_pairs]
    random.Random(seed).shuffle(negative_pairs)
    negative_edge_index = torch.tensor(negative_pairs[:len(train_transactions)], dtype=torch.long).T

    return GraphData(
        NODE_NAMES,
        train_transactions,
        observed_transactions,
        train_edge_index,
        train_edge_features,
        observed_edge_index,
        observed_edge_features,
        X,
        negative_edge_index,
    )
