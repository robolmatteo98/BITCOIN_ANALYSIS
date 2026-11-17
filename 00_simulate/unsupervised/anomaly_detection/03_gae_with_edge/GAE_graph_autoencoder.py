"""Entry point dell'esperimento GAE con feature sugli archi."""

import torch

try:
    from .gae_autoencoder.GAT_model import GraphAutoencoder as GATEdgeGraphAutoencoder
    from .gae_autoencoder.GCN_model import GraphAutoencoder as GCNEdgeGraphAutoencoder
    from .gae_autoencoder.analysis import calculate_anomaly_edge
    from .gae_autoencoder.config import TEST_SPLITS, set_seed
    from .gae_autoencoder.data import prepare_graph
    from .gae_autoencoder.training import train_model
except ImportError:
    # Consente anche l'esecuzione diretta dello script.
    from gae_autoencoder.GAT_model import GraphAutoencoder as GATEdgeGraphAutoencoder
    from gae_autoencoder.GCN_model import GraphAutoencoder as GCNEdgeGraphAutoencoder
    from gae_autoencoder.analysis import calculate_anomaly_edge
    from gae_autoencoder.config import TEST_SPLITS, set_seed
    from gae_autoencoder.data import prepare_graph
    from gae_autoencoder.training import train_model


def main() -> None:
    print("\n" + "=" * 65)
    print("CONFRONTO GCN vs GAT - GAE EDGE-AWARE")
    print("=" * 65)

    for model_name, model_class in (
        ("GCN", GCNEdgeGraphAutoencoder),
        ("GAT", GATEdgeGraphAutoencoder),
    ):
        print(f"\n{'-' * 65}\nMODELLO {model_name}\n{'-' * 65}")
        for experiment_id, test_transactions in enumerate(TEST_SPLITS, start=1):
            set_seed(42 + experiment_id)
            data = prepare_graph(seed=42 + experiment_id, test_transactions=test_transactions)
            model = train_model(data, model_class=model_class)
            model.eval()

            with torch.no_grad():
                _, logits = model(data.X, data.P, data.edge_features)
                probabilities = torch.sigmoid(logits)

            print(f"\nEsperimento {experiment_id}")
            print("Risultati sugli archi di test:")
            for src, dst in data.test_edges:
                print(
                    f"  {data.node_names[src]} -> {data.node_names[dst]} | "
                    f"Logit: {logits[src, dst].item():.6f} | "
                    f"Probabilità: {probabilities[src, dst].item():.6f} | "
                    f"Anomaly score: {calculate_anomaly_edge(probabilities[src, dst]):.6f}"
                )


if __name__ == "__main__":
    main()
