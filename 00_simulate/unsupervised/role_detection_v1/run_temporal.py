"""Esegue la role detection v1 separando storico e finestra temporale."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
UNSUPERVISED_ROOT = PACKAGE_ROOT.parent
if str(UNSUPERVISED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSUPERVISED_ROOT))

from role_detection.kmeans import kmeans_fit, kmeans_predict  # noqa: E402
from role_detection.gae_autoencoder.config import set_seed  # noqa: E402
from role_detection.gae_autoencoder.training import train_model  # noqa: E402

from .analysis import assign_roles, describe_clusters  # noqa: E402
from .data_from_db import load_scoped_features, load_v1_dataset  # noqa: E402
from .graph_builder import build_graph_data  # noqa: E402


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
        help="numero di cluster finali, incluso coinbase-like (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path("07_reports/role_detection_v1_temporal.json"),
        help="report JSON della pipeline temporale",
    )
    return parser.parse_args()


def _temporal_assignments(
    data,
    embeddings: torch.Tensor,
    clusters: int,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Crea il cluster coinbase e assegna gli altri nodi ai centroidi storici."""
    coinbase_count = int(data.coinbase_nodes.sum())
    regular_indices = torch.where(data.coinbase_nodes == False)[0]

    if not len(regular_indices) and not coinbase_count:
        raise ValueError("la finestra temporale non contiene nodi utilizzabili")
    if coinbase_count and clusters < 2 and len(regular_indices):
        raise ValueError("con nodi coinbase e nodi normali servono almeno 2 cluster")

    # -1 identifica il gruppo esplicito degli indirizzi che hanno ricevuto
    # ricompense coinbase; non viene incluso nell'assegnazione ai centroidi.
    assignments = torch.full(
        (len(data.node_names),), -1, dtype=torch.long, device=embeddings.device
    )
    if len(regular_indices):
        assignments[regular_indices] = kmeans_predict(
            embeddings[regular_indices], centroids
        )
    return assignments


def main() -> int:
    args = parse_args()
    if args.limit < 1 or args.history_limit < 1 or args.clusters < 1:
        raise ValueError("limit, history-limit e clusters devono essere positivi")

    set_seed(args.seed)

    print("Caricamento storico e finestra di analisi dal database...", flush=True)
    all_flows, _ = load_v1_dataset(
        args.database,
        args.limit,
        history_limit=args.history_limit,
        include_features=False,
    )
    transaction_ids = sorted(
        all_flows.transaction_id.dropna().astype(int).unique().tolist()
    )
    history_ids = transaction_ids[:args.history_limit]
    analysis_ids = transaction_ids[args.history_limit:]

    if not history_ids or not analysis_ids:
        raise ValueError("servono sia transazioni storiche sia transazioni nella finestra di analisi")

    # Separa fisicamente i flussi: il grafo storico non vede la finestra futura.
    history_flows = all_flows[all_flows["transaction_id"].isin(history_ids)].copy()
    analysis_flows = all_flows[all_flows["transaction_id"].isin(analysis_ids)].copy()
    history_features = load_scoped_features(args.database, history_ids)
    analysis_features = load_scoped_features(args.database, analysis_ids)

    # Graphs Data
    history_data = build_graph_data(
        history_flows, history_features, seed=args.seed
    )
    analysis_data = build_graph_data(
        analysis_flows, analysis_features, seed=args.seed
    )

    print(f"Transazioni storiche: {len(history_ids)}")
    print(f"Transazioni nella finestra: {len(analysis_ids)}")
    print(f"Nodi storici: {len(history_data.node_names)}")
    print(f"Nodi nella finestra: {len(analysis_data.node_names)}")
    print("Nuovi indirizzi nella finestra: " f"{len(set(analysis_data.node_names) - set(history_data.node_names))}")
    print(f"Feature per nodo: {len(history_data.feature_names)}")
    print("\nTraining esclusivamente sullo storico...")

    # Il modello impara i pesi solo dal grafo storico.
    model = train_model(history_data)
    model.eval()

    # I pesi appresi sullo storico vengono applicati sia allo storico sia al grafo della finestra. 
    # Questo permette di calcolare embedding anche per indirizzi nuovi.
    with torch.no_grad():
        history_embeddings = model.encode(history_data.X, history_data.P)
        analysis_embeddings = model.encode(analysis_data.X, analysis_data.P)

    # trova indici storici che non sono coinbase
    history_regular_indices = torch.where(history_data.coinbase_nodes == False)[0]

    # Il numero di centroidi dipende dai cluster che dovranno essere restituiti
    # nella finestra: se la finestra contiene coinbase, uno dei cluster è riservato a coinbase-like.
    regular_clusters = (
        args.clusters - 1
        if int(analysis_data.coinbase_nodes.sum())
        else args.clusters
    )
    if regular_clusters < 1:
        raise ValueError("serve almeno un cluster per gli indirizzi non coinbase")
    if regular_clusters > len(history_regular_indices):
        raise ValueError("i cluster superano il numero di nodi non coinbase storici")
    _, historical_centroids = kmeans_fit(
        history_embeddings[history_regular_indices],
        n_clusters=regular_clusters,
    )

    print(f"Embedding finestra: {tuple(analysis_embeddings.shape)}")
    assignments = _temporal_assignments(
        analysis_data, analysis_embeddings, args.clusters, historical_centroids
    )
    describe_clusters(analysis_data, assignments)
    roles = assign_roles(analysis_data, assignments)

    print("\nRuoli assegnati nella finestra:")
    for cluster_id in sorted(roles):
        result = roles[cluster_id]
        print(f"  Cluster {cluster_id}: {result['role']} ({result['size']} indirizzi)")
        print(f"    Wallet rappresentativo: {result['representative']}")

    report = {
        "experiment": {
            "database": args.database,
            "mode": "temporal",
            "history_transaction_ids": history_ids,
            "analysis_transaction_ids": analysis_ids,
            "history_limit": args.history_limit,
            "analysis_limit": args.limit,
            "seed": args.seed,
            "clusters": args.clusters,
            "history_num_nodes": len(history_data.node_names),
            "analysis_num_nodes": len(analysis_data.node_names),
            "new_analysis_nodes": len(
                set(analysis_data.node_names) - set(history_data.node_names)
            ),
            "analysis_num_edges": int(analysis_data.positive_edges.shape[1]),
            "analysis_coinbase_nodes": int(analysis_data.coinbase_nodes.sum()),
            "embedding_dim": int(analysis_embeddings.shape[1]),
        },
        "analysis_nodes": analysis_data.node_names,
        "analysis_embeddings": analysis_embeddings.cpu().tolist(),
        "assignments": assignments.cpu().tolist(),
        "roles": {str(cluster_id): result for cluster_id, result in roles.items()},
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