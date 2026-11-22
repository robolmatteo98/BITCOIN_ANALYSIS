"""Role detection temporale della pipeline base.

Il modello viene addestrato esclusivamente sullo storico; gli indirizzi della
finestra successiva vengono poi assegnati ai centroidi appresi sullo storico.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
UNSUPERVISED_ROOT = PACKAGE_ROOT.parent.parent
if str(UNSUPERVISED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSUPERVISED_ROOT))

from kmeans import kmeans_fit, kmeans_predict  # noqa: E402
from kmeans.print_graph import (  # noqa: E402
    plot_clustered_graph,
    plot_embedding_clusters,
)
from measure_algo import silhouette_score  # noqa: E402
from role_detection.gae_autoencoder.config import set_seed  # noqa: E402
from role_detection.base.data_from_db import load_transactions  # noqa: E402
from role_detection.base.graph_builder import build_graph_data  # noqa: E402
from role_detection.gae_autoencoder.training import train_model  # noqa: E402
from kmeans.analysis.cluster_analysis import (  # noqa: E402
    describe_clusters,
    interpret_clusters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", help="database PostgreSQL; default: DB_NAME")
    parser.add_argument(
        "--limit", type=int, default=500,
        help="numero di transazioni nella finestra di analisi (default: 500)",
    )
    parser.add_argument(
        "--history-limit", type=int, default=1000,
        help="numero di transazioni storiche per il training (default: 1000)",
    )
    parser.add_argument(
        "--clusters", type=int, default=4,
        help="numero di cluster (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture", choices=("gcn", "gat"), default="gcn",
        help="encoder da usare (default: gcn)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("07_reports/role_detection_temporal.json"),
        help="report JSON della pipeline temporale",
    )
    parser.add_argument(
        "--graph-output", type=Path,
        default=Path("07_reports/role_detection_temporal_graph.png"),
    )
    parser.add_argument(
        "--cluster-output", type=Path,
        default=Path("07_reports/role_detection_temporal_clusters.png"),
    )
    return parser.parse_args()


def _split_transactions(rows, history_limit: int):
    """Divide le righe per transaction id mantenendo l'ordine temporale del DB."""
    transaction_ids = list(rows["transaction_id"].dropna().drop_duplicates())
    history_ids = transaction_ids[:history_limit]
    analysis_ids = transaction_ids[history_limit:]
    history = rows[rows["txid"].isin(history_ids)].copy()
    analysis = rows[rows["txid"].isin(analysis_ids)].copy()
    return history, analysis, history_ids, analysis_ids


def _build_data(rows, seed: int):
    transactions = [
        (row.from_address, row.to_address, row.flow_amount, row.time)
        for row in rows.itertuples(index=False)
    ]
    return build_graph_data(transactions, seed=seed)


def main() -> int:
    args = parse_args()
    if args.limit < 1 or args.history_limit < 1 or args.clusters < 1:
        raise ValueError("limit, history-limit e clusters devono essere positivi")

    set_seed(args.seed)

    print("Caricamento storico e finestra di analisi dal database...", flush=True)
    rows = load_transactions(
        database=args.database,
        limit=args.history_limit + args.limit,
    )
    if rows.empty:
        raise ValueError("il database non contiene flussi address -> address")

    history_rows, analysis_rows, history_ids, analysis_ids = _split_transactions(
        rows, args.history_limit
    )
    if history_rows.empty or analysis_rows.empty:
        raise ValueError("servono sia transazioni storiche sia transazioni nella finestra")

    history_data = _build_data(history_rows, args.seed)
    analysis_data = _build_data(analysis_rows, args.seed)

    print(f"Transazioni storiche: {len(history_ids)}")
    print(f"Transazioni nella finestra: {len(analysis_ids)}")
    print(f"Nodi storici: {len(history_data.node_names)}")
    print(f"Nodi nella finestra: {len(analysis_data.node_names)}")
    print("Training esclusivamente sullo storico...")

    model = train_model(history_data, architecture=args.architecture)
    model.eval()
    with torch.no_grad():
        history_embeddings = model.encode(history_data.X, history_data.P)
        analysis_embeddings = model.encode(analysis_data.X, analysis_data.P)

    if args.clusters > len(history_data.node_names):
        raise ValueError("i cluster superano il numero di nodi storici")

    _, centroids = kmeans_fit(
        history_embeddings,
        n_clusters=args.clusters,
    )
    assignments = kmeans_predict(analysis_embeddings, centroids)

    graph_output = plot_clustered_graph(
        analysis_data,
        assignments,
        args.graph_output,
        title="Role detection temporale - grafo clusterizzato",
        seed=args.seed,
    )
    cluster_output = plot_embedding_clusters(
        analysis_embeddings,
        assignments,
        args.cluster_output,
        title="Role detection temporale - cluster degli embedding",
    )
    silhouette = silhouette_score(analysis_embeddings, assignments)

    print(f"Grafo clusterizzato salvato in: {graph_output}")
    print(f"Visualizzazione cluster salvata in: {cluster_output}")
    print(f"Silhouette score finestra: {silhouette:.4f}")
    describe_clusters(analysis_data, assignments)
    roles = interpret_clusters(analysis_data, assignments)

    print("\nWallet della finestra e cluster assegnato:")
    for name, cluster_id in zip(analysis_data.node_names, assignments.tolist()):
        print(f"  Wallet {name} -> Cluster {cluster_id}")

    report = {
        "experiment": {
            "database": args.database,
            "mode": "temporal",
            "history_transaction_ids": history_ids,
            "analysis_transaction_ids": analysis_ids,
            "history_limit": args.history_limit,
            "analysis_limit": args.limit,
            "seed": args.seed,
            "architecture": args.architecture,
            "clusters": args.clusters,
            "history_nodes": len(history_data.node_names),
            "analysis_nodes": len(analysis_data.node_names),
            "silhouette_score": silhouette,
        },
        "history": {
            "nodes": history_data.node_names,
            "embeddings": history_embeddings.detach().cpu().tolist(),
            "centroids": centroids.detach().cpu().tolist(),
        },
        "analysis": {
            "nodes": analysis_data.node_names,
            "embeddings": analysis_embeddings.detach().cpu().tolist(),
            "assignments": assignments.detach().cpu().tolist(),
            "roles": {str(cluster_id): role for cluster_id, role in roles.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport salvato in: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
