"""Scoperta non supervisionata dei ruoli dei wallet con GCN + clustering."""

import sys
from pathlib import Path

import torch


UNSUPERVISED_ROOT = Path(__file__).resolve().parent.parent
if str(UNSUPERVISED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSUPERVISED_ROOT))

from kmeans import kmeans_assignments  # noqa: E402
from kmeans.analysis.cluster_analysis import (  # noqa: E402
    describe_clusters,
    interpret_clusters,
)
from role_detection.gae_autoencoder.config import (  # noqa: E402
    NODE_NAMES,
    TRANSACTIONS,
    set_seed,
)
from role_detection.gae_autoencoder.graph_builder import build_graph_data  # noqa: E402
from role_detection.gae_autoencoder.training import train_model  # noqa: E402


def main() -> None:
    print("\n" + "=" * 72)
    print("ROLE DETECTION UNSUPERVISED - GCN SU GRAFO STATICO")
    print("=" * 72)
    print("Le etichette dei cluster sono descrittive, non ground truth supervisionato.")

    set_seed(42)

    # Crea il grafo GAE
    data = build_graph_data(TRANSACTIONS, node_names=NODE_NAMES, seed=42)

    model = train_model(data)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.X, data.P)

    assignments = kmeans_assignments(embeddings, n_clusters=4)
    describe_clusters(data, assignments)
    interpret_clusters(data, assignments)

    print("\nWallet e cluster assegnato:")
    for name, cluster_id in zip(data.node_names, assignments):
        print(f"  Wallet {name} -> Cluster {cluster_id}")

    print("\nNota:")
    print("  I ruoli sono euristici e descrittivi, non etichette supervisionate.")


if __name__ == "__main__":
    main()
