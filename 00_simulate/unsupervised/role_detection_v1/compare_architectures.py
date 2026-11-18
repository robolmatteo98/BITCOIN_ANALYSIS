"""Confronta GCN e GAT sulla stessa pipeline temporale v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PACKAGE_ROOT = Path(__file__).resolve().parent
UNSUPERVISED_ROOT = PACKAGE_ROOT.parent
if str(UNSUPERVISED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSUPERVISED_ROOT))

from kmeans import kmeans_fit, kmeans_predict  # noqa: E402
from role_detection.gae_autoencoder.config import set_seed  # noqa: E402
from role_detection.gae_autoencoder.training import train_model  # noqa: E402

from kmeans.analysis.cluster_analysis_v1 import assign_roles  # noqa: E402
from .data_from_db import load_scoped_features, load_v1_dataset  # noqa: E402
from .graph_builder import build_graph_data  # noqa: E402
from .verify_stability import _distance_correlation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", help="database PostgreSQL; default: DB_NAME")
    parser.add_argument("--history-limit", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path("07_reports/role_detection_v1_gcn_vs_gat.json"),
    )
    return parser.parse_args()


def _assign_temporal_clusters(data, embeddings, centroids):
    assignments = torch.full(
        (len(data.node_names),), -1, dtype=torch.long, device=embeddings.device
    )
    regular_mask = ~data.coinbase_nodes
    if regular_mask.any():
        assignments[regular_mask] = kmeans_predict(
            embeddings[regular_mask], centroids
        )
    return assignments


def _run_architecture(
    architecture,
    history_flows,
    analysis_flows,
    history_features,
    analysis_features,
    args,
):
    set_seed(args.seed)
    history_data = build_graph_data(history_flows, history_features, seed=args.seed)
    analysis_data = build_graph_data(analysis_flows, analysis_features, seed=args.seed)
    model = train_model(history_data, architecture=architecture)
    model.eval()

    with torch.no_grad():
        history_embeddings = model.encode(history_data.X, history_data.P)
        analysis_embeddings = model.encode(analysis_data.X, analysis_data.P)
        train_pairs = torch.cat(
            [history_data.positive_edges, history_data.negative_edges], dim=1
        )
        train_logits = model.score(history_embeddings, train_pairs)
        train_loss = F.binary_cross_entropy_with_logits(
            train_logits, history_data.labels
        )
        predictions = (torch.sigmoid(train_logits) >= 0.5).float()
        train_accuracy = (predictions == history_data.labels).float().mean()
        positive_score = model.score(
            history_embeddings, history_data.positive_edges
        ).mean()
        negative_score = model.score(
            history_embeddings, history_data.negative_edges
        ).mean()

    history_regular = ~history_data.coinbase_nodes
    regular_clusters = (
        args.clusters - 1
        if bool(analysis_data.coinbase_nodes.any())
        else args.clusters
    )
    _, centroids = kmeans_fit(
        history_embeddings[history_regular],
        n_clusters=regular_clusters,
    )
    assignments = _assign_temporal_clusters(
        analysis_data, analysis_embeddings, centroids
    )
    roles = assign_roles(analysis_data, assignments)
    cluster_sizes = {
        str(cluster_id): int((assignments == cluster_id).sum())
        for cluster_id in sorted(set(assignments.tolist()))
    }
    role_counts = {}
    role_node_counts = {}
    for role_data in roles.values():
        role_counts[role_data["role"]] = role_counts.get(role_data["role"], 0) + 1
        role_node_counts[role_data["role"]] = (
            role_node_counts.get(role_data["role"], 0) + role_data["size"]
        )

    return {
        "architecture": architecture,
        "training_reconstruction": {
            "loss": float(train_loss),
            "accuracy": float(train_accuracy),
            "positive_score_mean": float(positive_score),
            "negative_score_mean": float(negative_score),
        },
        "embedding": {
            "history_shape": list(history_embeddings.shape),
            "analysis_shape": list(analysis_embeddings.shape),
            "history_mean_norm": float(history_embeddings.norm(dim=1).mean()),
            "analysis_mean_norm": float(analysis_embeddings.norm(dim=1).mean()),
        },
        "analysis": {
            "cluster_sizes": cluster_sizes,
            "role_counts": role_counts,
            "role_node_counts": role_node_counts,
            "roles": {str(cluster_id): result for cluster_id, result in roles.items()},
        },
        "_history_embeddings": history_embeddings,
        "_analysis_embeddings": analysis_embeddings,
        "_assignments": assignments,
    }


def main() -> int:
    args = parse_args()
    print("Caricamento dello stesso split per GCN e GAT...", flush=True)
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
        raise ValueError("servono sia transazioni storiche sia transazioni di analisi")

    history_flows = all_flows[
        all_flows["transaction_id"].isin(history_ids)
    ].copy()
    analysis_flows = all_flows[
        all_flows["transaction_id"].isin(analysis_ids)
    ].copy()
    history_features = load_scoped_features(args.database, history_ids)
    analysis_features = load_scoped_features(args.database, analysis_ids)

    results = {}
    for architecture in ("gcn", "gat"):
        print(f"\nTraining {architecture.upper()}...", flush=True)
        results[architecture] = _run_architecture(
            architecture,
            history_flows,
            analysis_flows,
            history_features,
            analysis_features,
            args,
        )

    gcn = results["gcn"]
    gat = results["gat"]
    gcn_pairs = gcn["_assignments"][:, None] == gcn["_assignments"][None, :]
    gat_pairs = gat["_assignments"][:, None] == gat["_assignments"][None, :]
    upper = torch.triu(torch.ones_like(gcn_pairs, dtype=torch.bool), diagonal=1)
    comparison = {
        "history_embedding_distance_correlation": _distance_correlation(
            gcn["_history_embeddings"], gat["_history_embeddings"]
        ),
        "analysis_embedding_distance_correlation": _distance_correlation(
            gcn["_analysis_embeddings"], gat["_analysis_embeddings"]
        ),
        "same_analysis_cluster_pair_agreement": float(
            (gcn_pairs[upper] == gat_pairs[upper]).float().mean()
        ),
    }

    for result in results.values():
        result.pop("_history_embeddings")
        result.pop("_analysis_embeddings")
        result.pop("_assignments")

    report = {
        "experiment": {
            "mode": "gcn_vs_gat",
            "database": args.database,
            "history_limit": args.history_limit,
            "analysis_limit": args.limit,
            "clusters": args.clusters,
            "seed": args.seed,
            "history_transaction_ids": history_ids,
            "analysis_transaction_ids": analysis_ids,
        },
        "models": results,
        "comparison": comparison,
        "interpretation": {
            "training_loss": "più basso indica una migliore ricostruzione sugli archi usati nel training",
            "positive_score": "più alto è preferibile",
            "negative_score": "più basso è preferibile",
            "distance_correlation": "misura la somiglianza della geometria degli embedding",
            "cluster_pair_agreement": "confronta la struttura dei cluster senza dipendere dai numeri assegnati",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nReport comparativo salvato in: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
