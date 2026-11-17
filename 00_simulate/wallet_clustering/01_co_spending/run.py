"""Esegue il co-spending su un piccolo db statico."""

from __future__ import annotations
from algorithm import find_clusters
from example_data import example_inputs

def main() -> int:

    # Carico i dati
    inputs = example_inputs()

    # Creo i clusters
    clusters = find_clusters(inputs)

    print(f"Transazioni: {inputs.tx_id.nunique() if not inputs.empty else 0}")
    print(f"Indirizzi sorgente: {inputs.source.nunique() if not inputs.empty else 0}")
    print(f"Cluster: {len(clusters)}")
    for cluster_id, cluster in enumerate(clusters, start=1):
        print(f"Cluster {cluster_id} ({len(cluster)} indirizzi): {', '.join(sorted(cluster))}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())