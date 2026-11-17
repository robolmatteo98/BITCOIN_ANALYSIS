"""Scoperta non supervisionata dei ruoli dei wallet con GCN + clustering."""

import torch

try:
    from .kmeans import kmeans_assignments
    from .cluster_analysis import describe_clusters, interpret_clusters
    from .gae_autoencoder.config import set_seed
    from .gae_autoencoder.data import prepare_graph
    from .gae_autoencoder.training import train_model
except ImportError:
    from kmeans import kmeans_assignments
    from cluster_analysis import describe_clusters, interpret_clusters
    from gae_autoencoder.config import set_seed
    from gae_autoencoder.data import prepare_graph
    from gae_autoencoder.training import train_model


def main() -> None:
    print("\n" + "=" * 72)
    print("ROLE DETECTION UNSUPERVISED - GCN SU GRAFO STATICO")
    print("=" * 72)
    print("Le etichette dei cluster sono descrittive, non ground truth supervisionato.")

    set_seed(42)

    # Crea il grafo GAE
    data = prepare_graph(seed=42)

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
