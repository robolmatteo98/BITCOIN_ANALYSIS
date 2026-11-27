"""Esegue la role detection v1 usando feature aggregate dal database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PACKAGE_ROOT = Path(__file__).resolve().parent
UNSUPERVISED_ROOT = PACKAGE_ROOT.parent.parent
if str(UNSUPERVISED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSUPERVISED_ROOT))

from kmeans import kmeans_assignments  # noqa: E402
from role_detection.gae_autoencoder.config import set_seed  # noqa: E402
from role_detection.gae_autoencoder.training import train_model  # noqa: E402
from kmeans.print_graph import (  # noqa: E402
    plot_clustered_graph,
    plot_embedding_clusters,
)
from measure_algo import silhouette_score  # noqa: E402

from kmeans.analysis.cluster_analysis_v1 import assign_roles, describe_v1_clusters  # noqa: E402
from .data_from_db import (  # noqa: E402
    aggregate_wallet_graph_inputs,
    load_v1_dataset,
)
from .graph_builder import build_graph_data  # noqa: E402
from role_detection.report_paths import (  # noqa: E402
    analysis_artifact_path,
    normalize_graph_mode,
    split_report_paths,
    write_split_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="numero di transazioni da caricare (default: 1000)",
    )
    parser.add_argument(
        "--history-limit", type=int, default=1000,
        help="transazioni precedenti usate come contesto (default: 1000)",
    )
    parser.add_argument(
        "--clusters", type=int, default=4,
        help="numero di cluster K-Means (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--graph-mode",
        choices=("address", "address_wallet_feature", "address_wallet", "wallet"),
        default="address_wallet_feature",
        help=(
            "grafo address, address con feature wallet oppure wallet "
            "(default: address_wallet_feature)"
        ),
    )
    parser.add_argument(
        "--architecture", choices=("gcn", "gat"), default="gcn",
        help="encoder da usare (default: gcn)",
    )
    parser.add_argument(
        "--matrix-format", choices=("dense", "sparse"), default="dense",
        help="formato della matrice di propagazione (default: dense)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="override opzionale; senza override salva in mode/history e mode/analysis",
    )
    parser.add_argument(
        "--graph-output", type=Path, default=None,
        help="PNG del grafo colorato per cluster",
    )
    parser.add_argument(
        "--cluster-output", type=Path, default=None,
        help="PNG della proiezione 2D degli embedding clusterizzati",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = split_report_paths("v1", args.graph_mode, "report.json")
    if args.output is not None:
        report_paths = {"analysis": args.output}
    args.graph_output = args.graph_output or analysis_artifact_path(
        "v1", args.graph_mode, "graph.png"
    )
    args.cluster_output = args.cluster_output or analysis_artifact_path(
        "v1", args.graph_mode, "clusters.png"
    )
    args.graph_mode = normalize_graph_mode(args.graph_mode)
    database = None  # Il loader legge DB_NAME e le credenziali dal .env.
    if args.limit < 1 or args.history_limit < 0 or args.history_limit > args.limit or args.clusters < 1:
        raise ValueError("limit e clusters devono essere positivi e history-limit non può superare limit")

    set_seed(args.seed)

    # Data Frames di flusso e feature. Il livello del grafo viene scelto dopo
    # il caricamento, così il database resta una sorgente unica.
    print("Caricamento transazioni e feature dal database...", flush=True)
    flows, address_features = load_v1_dataset(
        database,
        args.limit,
        history_limit=args.history_limit,
        include_wallet_features=args.graph_mode != "address",
    )

    if args.graph_mode == "wallet":
        # Nel wallet mode cambiano sia gli archi sia le feature: il costruttore
        # riceve già i DataFrame aggregati e crea un vero wallet -> wallet.
        flows, address_features = aggregate_wallet_graph_inputs(
            flows, address_features, split="analysis"
        )

    # Il costruttore v1 crea la matrice P e le feature del livello richiesto.
    data = build_graph_data(
        flows,
        address_features,
        seed=args.seed,
        graph_mode=args.graph_mode,
        matrix_format=args.matrix_format,
    )

    coinbase_count = int(data.coinbase_nodes.sum())
    regular_indices = torch.where(~data.coinbase_nodes)[0]
    if not len(regular_indices) and not coinbase_count:
        raise ValueError("il grafo non contiene nodi utilizzabili")
    if coinbase_count and args.clusters < 2 and len(regular_indices):
        raise ValueError(
            "con nodi coinbase e nodi normali servono almeno 2 cluster: "
            "uno coinbase-like e almeno uno K-Means"
        )
    regular_clusters = args.clusters - 1 if coinbase_count else args.clusters
    if regular_clusters > len(regular_indices):
        raise ValueError("i cluster K-Means superano il numero di nodi non coinbase")

    transaction_ids = sorted(flows.transaction_id.dropna().astype(int).unique().tolist())
    analysis_start = min(args.history_limit, len(transaction_ids))
    history_ids = transaction_ids[:analysis_start]
    analysis_ids = transaction_ids[analysis_start:]

    print(f"Cutoff totale transazioni: {args.limit}")
    print(f"Contesto storico richiesto: {args.history_limit}")
    print(f"Transazioni totali caricate: {len(transaction_ids)}")
    print(f"Transazioni di contesto: {len(history_ids)}")
    print(f"Transazioni di analisi: {len(analysis_ids)}")
    print(f"Flussi caricati: {len(flows)}")
    print(f"Righe coinbase nel campione: {int(flows['is_coinbase'].fillna(False).sum())}")
    print(f"Livello grafo: {args.graph_mode}")
    print(f"Nodi: {len(data.node_names)}")
    print(f"Nodi coinbase: {coinbase_count}")
    print(f"Feature per nodo: {len(data.feature_names)}")
    print(f"Archi positivi distinti: {data.positive_edges.shape[1]}")
    print(f"Matrice X: {tuple(data.X.shape)}")
    print(f"Matrice P: {tuple(data.P.shape)}")
    print(
        "Esempio nodo 0: "
        f"{data.node_names[0]} -> feature standardizzate "
        f"{data.X[0, :5].tolist()} ..."
    )
    print("\nFeature utilizzate:")
    print("  " + ", ".join(data.feature_names))
    print("\nAvvio training role detection v1...")

    # Riutilizzo il GAE della baseline: cambia la rappresentazione dei nodi, non l'architettura di apprendimento, così il confronto resta controllato.
    model = train_model(data, architecture=args.architecture)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.X, data.P)

        print(f"Embedding: {tuple(embeddings.shape)}")
        print(
            "Esempio embedding nodo 0: "
            f"{embeddings[0, :5].tolist()} ..."
        )

        # Controllo quantitativo della ricostruzione sugli stessi archi usati
        # dal training. Non è ancora una metrica di generalizzazione.
        train_pairs = torch.cat([data.positive_edges, data.negative_edges], dim=1)
        train_logits = model.score(embeddings, train_pairs)
        train_loss = F.binary_cross_entropy_with_logits(train_logits, data.labels)
        train_predictions = (torch.sigmoid(train_logits) >= 0.5).float()
        train_accuracy = (train_predictions == data.labels).float().mean()
        positive_score = model.score(embeddings, data.positive_edges).mean()
        negative_score = model.score(embeddings, data.negative_edges).mean()

    # I nodi coinbase hanno una semantica specifica e non vengono mescolati
    # ai ruoli appresi dal K-Means sugli archi address → address.
    assignments = torch.full(
        (len(data.node_names),), -1, dtype=torch.long, device=embeddings.device
    )
    if len(regular_indices):
        regular_embeddings = embeddings[regular_indices]
        regular_assignments = kmeans_assignments(
            regular_embeddings, n_clusters=regular_clusters
        )
        assignments[regular_indices] = regular_assignments
    graph_output = plot_clustered_graph(
        data,
        assignments,
        args.graph_output,
        title="Role detection v1 - grafo clusterizzato",
        seed=args.seed,
    )
    print(f"Grafo clusterizzato salvato in: {graph_output}")
    cluster_output = plot_embedding_clusters(
        embeddings,
        assignments,
        args.cluster_output,
        title="Role detection v1 - cluster degli embedding",
    )
    print(f"Visualizzazione cluster salvata in: {cluster_output}")
    silhouette = silhouette_score(embeddings, assignments)
    print(f"Silhouette score: {silhouette:.4f}")
    describe_v1_clusters(data, assignments)
    roles = assign_roles(data, assignments)

    print("\nRuoli assegnati:")
    for cluster_id in sorted(roles):
        result = roles[cluster_id]
        print(f"  Cluster {cluster_id}: {result['role']} ({result['size']} indirizzi)")
        print(f"    Wallet rappresentativo: {result['representative']}")

    report = {
        "experiment": {
            "database": "from_env",
            "history_limit": args.history_limit,
            "analysis_limit": len(analysis_ids),
            "history_transaction_ids": history_ids,
            "analysis_transaction_ids": analysis_ids,
            "seed": args.seed,
            "architecture": args.architecture,
            "graph_mode": args.graph_mode,
            "clusters": args.clusters,
            "feature_names": data.feature_names,
            "num_nodes": len(data.node_names),
            "num_edges": int(data.positive_edges.shape[1]),
            "coinbase_rows": int(flows["is_coinbase"].fillna(False).sum()),
            "embedding_dim": int(embeddings.shape[1]),
            "silhouette_score": silhouette,
        },
        "training_reconstruction": {
            "loss": float(train_loss),
            "accuracy": float(train_accuracy),
            "positive_score_mean": float(positive_score),
            "negative_score_mean": float(negative_score),
        },
        "nodes": data.node_names,
        "embeddings": embeddings.detach().cpu().tolist(),
        "assignments": assignments.detach().cpu().tolist(),
        "roles": {str(cluster_id): result for cluster_id, result in roles.items()},
    }
    write_split_json(report, report_paths)
    print("\nReport salvati in:")
    for path in report_paths.values():
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
