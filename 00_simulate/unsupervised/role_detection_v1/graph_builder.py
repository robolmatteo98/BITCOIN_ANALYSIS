"""Costruzione del grafo e delle feature dei nodi per la role detection v1."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from .data import GraphData


FEATURE_NAMES = [
    "incoming_degree",
    "outgoing_degree",
    "incoming_counterparties",
    "outgoing_counterparties",
    "received_output_count",
    "coinbase_received_count",
    "coinbase_received_amount",
    "spent_output_count",
    "unspent_output_count",
    "total_received",
    "total_sent",
    "net_flow",
    "estimated_balance",
    "first_seen_days",
    "last_seen_days",
    "active_span_days",
    "active_days",
    "activity_count",
]


def _sample_negative_edges(
    num_nodes: int,
    positive_edges: set[tuple[int, int]],
    count: int,
    seed: int,
) -> torch.Tensor:
    """Campiona negativi senza creare tutte le coppie possibili."""
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


def _standardize(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applica log1p alle code lunghe e restituisce valori grezzi e standardizzati."""
    transformed = values.copy()

    # I conteggi, gli importi e le durate sono spesso molto sbilanciati.
    non_negative_columns = [
        "incoming_degree", "outgoing_degree", "incoming_counterparties",
        "outgoing_counterparties", "received_output_count",
        "coinbase_received_count", "coinbase_received_amount",
        "spent_output_count",
        "unspent_output_count", "total_received", "total_sent", "estimated_balance",
        "first_seen_days", "last_seen_days", "active_span_days", "active_days",
        "activity_count",
    ]
    for column in non_negative_columns:
        transformed[column] = np.log1p(transformed[column].clip(lower=0.0))

    # Manteniamo il segno del flusso netto, comprimendone la scala.
    transformed["net_flow"] = np.sign(transformed["net_flow"]) * np.log1p(
        np.abs(transformed["net_flow"])
    )

    means = transformed.mean(axis=0)
    stds = transformed.std(axis=0, ddof=0).replace(0, 1.0)
    standardized = (transformed - means) / stds
    return transformed, standardized


def build_graph_data(
    flows: pd.DataFrame,
    address_features: pd.DataFrame,
    seed: int = 42,
) -> GraphData:
    """Costruisce GraphData v1 usando feature SQL dello stesso campione."""

    # Controllo
    required_flow_columns = {"from_address", "to_address", "flow_amount", "time"}
    missing = required_flow_columns - set(flows.columns)
    if missing:
        raise ValueError(f"colonne mancanti nei flussi: {', '.join(sorted(missing))}")
    if flows.empty:
        raise ValueError("nessun flusso utilizzabile")
    if address_features is None or address_features.empty:
        raise ValueError(
            "mancano le feature scoped degli indirizzi: verificare "
            "address_feature_summary_for_transactions"
        )
    
    required_scoped_columns = {
        "address", "received_output_count", "spent_output_count",
        "coinbase_received_count", "coinbase_received_amount",
        "unspent_output_count", "total_received", "total_sent", "net_flow",
        "estimated_balance", "first_seen", "last_seen", "active_span",
        "activity_count", "active_days",
    }
    missing_scoped_columns = required_scoped_columns - set(address_features.columns)
    if missing_scoped_columns:
        raise ValueError(
            "colonne mancanti nelle feature scoped: "
            + ", ".join(sorted(missing_scoped_columns))
        )

    # Queste feature arrivano dalla funzione SQL parametrica e sono già
    # limitate agli stessi transaction_id del campione.
    scoped = address_features.copy()
    scoped["address"] = scoped["address"].astype(str)

    # I flussi coinbase vengono usati per le feature SQL, ma non sono archi
    # address → address e quindi vengono esclusi dalla costruzione del grafo.
    # converte flow_amount in numerico
    # converte time in numerico
    flows = flows.dropna(subset=["from_address", "to_address"]).copy()
    if flows.empty:
        raise ValueError("il campione contiene solo flussi coinbase o indirizzi non risolti")
    flows["flow_amount"] = pd.to_numeric(flows["flow_amount"], errors="coerce").fillna(0.0)
    flows["time"] = pd.to_numeric(flows["time"], errors="coerce").fillna(0.0)

    # Unisce tutti i from e to, con unique elimina i duplicati.
    # Aggiunge anche gli indirizzi presenti nelle feature SQL: in questo modo
    # un destinatario coinbase, che non ha un from_address, resta nel grafo
    # come nodo isolato e può essere riconosciuto dal modello.
    flow_addresses = pd.concat(
        [flows["from_address"], flows["to_address"]], ignore_index=True
    )
    addresses = pd.unique(
        pd.concat([flow_addresses, scoped["address"]], ignore_index=True)
    ).astype(str).tolist()
    node_to_id = {address: index for index, address in enumerate(addresses)}
    num_nodes = len(addresses)

    # Creo la matrice di adiacenza pesata con il numero di flussi osservati.
    # Somma 1 per ogni arco presente
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    positive_pairs: set[tuple[int, int]] = set()
    for row in flows.itertuples(index=False):
        source = node_to_id[str(row.from_address)]
        target = node_to_id[str(row.to_address)]
        adjacency[source, target] += 1.0
        positive_pairs.add((source, target))

    # Calcolo le feature strutturali ed economiche direttamente dagli archi
    # address → address del campione. Non uso la vista globale degli indirizzi:
    # in questo modo una prova sulle prime N transazioni non vede dati futuri.

    # Calcola il numero di flussi in uscita e in entrata raggruppando per indirizzo
    outgoing = flows.groupby("from_address").agg(
        outgoing_degree=("to_address", "size"),
        outgoing_counterparties=("to_address", "nunique"),
        total_sent=("flow_amount", "sum"),
    )
    incoming = flows.groupby("to_address").agg(
        incoming_degree=("from_address", "size"),
        incoming_counterparties=("from_address", "nunique"),
        total_received=("flow_amount", "sum"),
    )

    node_features = pd.DataFrame({"address": addresses})
    # reset_index() per riportarlo a una normale colonna
    # unisce le colonne di node_features con outgoing
    node_features = node_features.merge(
        outgoing.reset_index().rename(columns={"from_address": "address"}),
        on="address", how="left",
    )

    # unisce le colonne con incoming
    node_features = node_features.merge(
        incoming.reset_index().rename(columns={"to_address": "address"}),
        on="address", how="left",
    )

    # unisce le info strutturali con quelle economiche / temporali provenienti da SQL
    overlapping = [
        "received_output_count", "coinbase_received_count",
        "coinbase_received_amount", "spent_output_count", "unspent_output_count",
        "total_received", "total_sent", "net_flow", "estimated_balance",
        "first_seen", "last_seen", "active_span", "activity_count", "active_days",
    ]
    node_features = node_features.drop(
        columns=[column for column in overlapping if column in node_features],
        errors="ignore",
    ).merge(scoped, on="address", how="left")

    # Controllo che tutte le feature esistano e siano numeriche
    for column in FEATURE_NAMES:
        if column not in node_features:
            node_features[column] = 0.0
        node_features[column] = pd.to_numeric(
            node_features[column], errors="coerce"
        ).fillna(0.0)

    # converte e sostituisce    
    for column in ("first_seen", "last_seen", "activity_count", "active_days"):
        node_features[column] = pd.to_numeric(
            node_features[column], errors="coerce"
        ).fillna(0.0)

    # Calcolo il timestamp minimo e il timestamp massimo, divido poi per il numero di secondi in un giorno
    # Calcolo per quanto tempo è rimasto attivo
    start_time = float(flows["time"].min())
    node_features["first_seen_days"] = (
        node_features["first_seen"] - start_time
    ).clip(lower=0.0) / 86400.0
    node_features["last_seen_days"] = (
        node_features["last_seen"] - start_time
    ).clip(lower=0.0) / 86400.0
    node_features["active_span_days"] = node_features["active_span"].clip(lower=0.0) / 86400.0

    # Separa i nodi che hanno ricevuto ricompense coinbase. La maschera viene
    # salvata in GraphData e usata da run.py per creare il cluster dedicato.
    coinbase_nodes = torch.tensor(
        node_features["coinbase_received_count"].to_numpy() > 0,
        dtype=torch.bool,
    )

    # seleziona le feature
    raw_features = node_features[FEATURE_NAMES].copy()
    role_features, standardized = _standardize(raw_features)

    # P[dest, source]: propagazione GCN con self-loop e normalizzazione per riga.
    P = adjacency.T + torch.eye(num_nodes)
    P = P / P.sum(dim=1, keepdim=True)

    # Gli archi positivi rappresentano relazioni osservate; i negativi sono coppie non osservate usate dal graph autoencoder.
    positive_edges = torch.tensor(
        sorted(positive_pairs), dtype=torch.long
    ).T.contiguous()
    negative_edges = _sample_negative_edges(
        num_nodes, positive_pairs, positive_edges.shape[1], seed
    )
    labels = torch.cat([
        torch.ones(positive_edges.shape[1]),
        torch.zeros(negative_edges.shape[1]),
    ])

    return GraphData(
        node_names=addresses,
        feature_names=FEATURE_NAMES,
        X=torch.tensor(standardized.to_numpy(), dtype=torch.float32),
        P=P,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        labels=labels,
        role_features=torch.tensor(role_features.to_numpy(), dtype=torch.float32),
        coinbase_nodes=coinbase_nodes,
    )
