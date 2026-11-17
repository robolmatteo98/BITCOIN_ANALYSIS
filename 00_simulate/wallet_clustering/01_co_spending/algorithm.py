from __future__ import annotations
import pandas as pd

MIN_CLUSTER_SIZE = 2

def find_clusters(inputs: pd.DataFrame) -> list[set[str]]:
    clusters: list[set[str]] = []

    # Raggruppo per tx_id, che identifica univocamente ogni transazione.
    # Ad esempio: tx_11 --> A1, A2.
    for tx_id, group in inputs.groupby("tx_id")["source"]:

        # 1. Elimina valori nulli.
        # 2. Converte gli indirizzi in stringhe.
        # 3. Elimina duplicati usando set.
        addresses = {str(value) for value in group if pd.notna(value)}
        if not addresses:
            continue

        # Cerca tutti i cluster che condividono almeno un indirizzo con questa transazione.
        # cluster & addresses = intersezione tra i due insiemi
        overlapping = [cluster for cluster in clusters if cluster & addresses]

        # Rimuove da clusters tutti i cluster sovrapposti ad addresses
        clusters = [cluster for cluster in clusters if not cluster & addresses]
        # crea un nuovo insieme con gli indirizzi iniziali
        merged = set(addresses)

        # aggiunge a merged tutti gli altri indirizzi dei cluster sovrapposti
        for cluster in overlapping:
            merged.update(cluster)

        # aggiunge il nuovo cluster alla lista
        clusters.append(merged)

    # Elimina le componenti troppo piccole.
    # I cluster vengono ordinati:
    # 1. prima per dimensione decrescente;
    # 2. in caso di parità, alfabeticamente.
    return sorted(
        (cluster for cluster in clusters if len(cluster) >= MIN_CLUSTER_SIZE),
        key=lambda cluster: (-len(cluster), sorted(cluster)),
    )