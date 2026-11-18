"""Verifica la stabilità di embedding e cluster su più seed."""

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

from .data_from_db import load_scoped_features, load_v1_dataset  # noqa: E402
from .graph_builder import build_graph_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", help="database PostgreSQL; default: DB_NAME")
    parser.add_argument("--history-limit", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument(
        "--architecture", choices=("gcn", "gat"), default="gcn",
        help="encoder da verificare (default: gcn)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45],
        help="seed da confrontare (default: 42 43 44 45)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("07_reports/role_detection_v1_stability.json"),
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


def _reorder_by_names(
    embeddings: torch.Tensor,
    assignments: torch.Tensor | None,
    names: list[str],
    reference_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    indices = {name: index for index, name in enumerate(names)}
    common = [name for name in reference_names if name in indices]
    reordered_indices = torch.tensor(
        [indices[name] for name in common], dtype=torch.long
    )
    reordered_embeddings = embeddings[reordered_indices]
    if assignments is None:
        return reordered_embeddings, None
    return reordered_embeddings, assignments[reordered_indices]


def _align_procrustes(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Allinea il candidato compensando rotazioni e cambi di segno."""
    reference_centered = reference - reference.mean(dim=0, keepdim=True)
    candidate_centered = candidate - candidate.mean(dim=0, keepdim=True)
    left, _, right = torch.linalg.svd(
        candidate_centered.T @ reference_centered
    )
    rotation = left @ right
    aligned = candidate_centered @ rotation
    scale = reference_centered.square().mean().sqrt().clamp_min(1e-12)
    normalized_rmse = (
        (aligned - reference_centered).square().mean().sqrt() / scale
    )
    return aligned, float(normalized_rmse)


def _distance_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first_distances = torch.cdist(first, first)
    second_distances = torch.cdist(second, second)
    upper = torch.triu(
        torch.ones_like(first_distances, dtype=torch.bool), diagonal=1
    )
    first_values = first_distances[upper]
    second_values = second_distances[upper]
    if first_values.numel() < 2:
        return 1.0
    first_values = first_values - first_values.mean()
    second_values = second_values - second_values.mean()
    denominator = first_values.norm() * second_values.norm()
    if denominator == 0:
        return 1.0 if torch.allclose(first_values, second_values) else 0.0
    return float((first_values @ second_values) / denominator)


def _coassignment_agreement(first: torch.Tensor, second: torch.Tensor) -> float:
    first_same = first[:, None] == first[None, :]
    second_same = second[:, None] == second[None, :]
    upper = torch.triu(torch.ones_like(first_same, dtype=torch.bool), diagonal=1)
    if not upper.any():
        return 1.0
    return float((first_same[upper] == second_same[upper]).float().mean())


def main() -> int:
    args = parse_args()
    if len(args.seeds) < 2:
        raise ValueError("servono almeno due seed per una verifica di stabilità")

    print("Caricamento storico e finestra di analisi...", flush=True)
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

    runs = []
    for seed in args.seeds:
        print(f"\nSeed {seed}: training...", flush=True)
        set_seed(seed)
        history_data = build_graph_data(history_flows, history_features, seed=seed)
        analysis_data = build_graph_data(analysis_flows, analysis_features, seed=seed)
        model = train_model(history_data, architecture=args.architecture)
        model.eval()
        with torch.no_grad():
            history_embeddings = model.encode(history_data.X, history_data.P)
            analysis_embeddings = model.encode(analysis_data.X, analysis_data.P)
            history_pairs = torch.cat(
                [history_data.positive_edges, history_data.negative_edges], dim=1
            )
            logits = model.score(history_embeddings, history_pairs)
            reconstruction_loss = F.binary_cross_entropy_with_logits(
                logits, history_data.labels
            )

        history_regular = ~history_data.coinbase_nodes
        regular_clusters = (
            args.clusters - 1
            if bool(analysis_data.coinbase_nodes.any())
            else args.clusters
        )
        _, centroids = kmeans_fit(
            history_embeddings[history_regular], n_clusters=regular_clusters
        )
        assignments = _assign_temporal_clusters(
            analysis_data, analysis_embeddings, centroids
        )
        runs.append({
            "seed": seed,
            "history_names": history_data.node_names,
            "analysis_names": analysis_data.node_names,
            "history_embeddings": history_embeddings.detach(),
            "analysis_embeddings": analysis_embeddings.detach(),
            "assignments": assignments.detach(),
            "reconstruction_loss": float(reconstruction_loss),
        })

    reference = runs[0]
    comparisons = []
    for run in runs[1:]:
        reference_history = reference["history_embeddings"]
        candidate_history, _ = _reorder_by_names(
            run["history_embeddings"],
            None,
            run["history_names"],
            reference["history_names"],
        )
        aligned_history, history_rmse = _align_procrustes(
            reference_history, candidate_history
        )

        reference_analysis = reference["analysis_embeddings"]
        candidate_analysis, candidate_assignments = _reorder_by_names(
            run["analysis_embeddings"],
            run["assignments"],
            run["analysis_names"],
            reference["analysis_names"],
        )
        _, analysis_rmse = _align_procrustes(
            reference_analysis, candidate_analysis
        )
        _, reference_assignments = _reorder_by_names(
            reference_analysis,
            reference["assignments"],
            reference["analysis_names"],
            reference["analysis_names"],
        )
        comparisons.append({
            "reference_seed": reference["seed"],
            "candidate_seed": run["seed"],
            "history_embedding_procrustes_rmse": history_rmse,
            "analysis_embedding_procrustes_rmse": analysis_rmse,
            "history_distance_correlation": _distance_correlation(
                reference_history, aligned_history
            ),
            "analysis_distance_correlation": _distance_correlation(
                reference_analysis, candidate_analysis
            ),
            "analysis_coassignment_agreement": _coassignment_agreement(
                reference_assignments, candidate_assignments
            ),
        })

    report = {
        "experiment": {
            "mode": "temporal_stability",
            "database": args.database,
            "history_limit": args.history_limit,
            "analysis_limit": args.limit,
            "clusters": args.clusters,
            "architecture": args.architecture,
            "seeds": args.seeds,
            "history_num_nodes": len(reference["history_names"]),
            "analysis_num_nodes": len(reference["analysis_names"]),
        },
        "runs": [
            {
                "seed": run["seed"],
                "reconstruction_loss": run["reconstruction_loss"],
            }
            for run in runs
        ],
        "comparisons_to_first_seed": comparisons,
        "interpretation": {
            "procrustes_rmse": "più basso è più stabile",
            "distance_correlation": "più vicino a 1 è più stabile",
            "coassignment_agreement": (
                "più vicino a 1 è più stabile; non dipende dai numeri dei cluster"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nReport stabilità salvato in: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
