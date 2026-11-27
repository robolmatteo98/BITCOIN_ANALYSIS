"""Esegue la role detection v1 separando storico e finestra temporale."""

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
    load_scoped_features,
    load_wallet_manifest,
    load_v1_dataset,
    summarize_wallet_graph,
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
        "--limit", type=int, default=11000,
        help="cutoff totale delle transazioni (default: 11000)",
    )
    parser.add_argument(
        "--history-limit", type=int, default=10000,
        help="numero di transazioni storiche (default: 10000)",
    )
    parser.add_argument(
        "--clusters", type=int, default=None,
        help="usa un solo numero di cluster invece della scansione (default: scansione 4-10)",
    )
    parser.add_argument(
        "--cluster-min", type=int, default=4,
        help="primo numero di cluster provato nella scansione (default: 4)",
    )
    parser.add_argument(
        "--cluster-max", type=int, default=10,
        help="ultimo numero di cluster provato nella scansione (default: 10)",
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
        help="PNG del grafo della finestra, colorato per cluster",
    )
    parser.add_argument(
        "--cluster-output", type=Path, default=None,
        help="PNG della proiezione 2D degli embedding clusterizzati",
    )
    return parser.parse_args()


def _temporal_assignments(
    data,
    embeddings: torch.Tensor,
    clusters: int,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Assegna i nodi della finestra temporale ai cluster storici."""

    # conta quanti nodi hanno ricevuto almeno una ricompensa coinbase
    coinbase_count = int(data.coinbase_nodes.sum())

    # ottiene gli indici dei nodi normali, non coinbase
    regular_indices = torch.where(data.coinbase_nodes == False)[0]

    # controlli
    if not len(regular_indices) and not coinbase_count:
        raise ValueError("la finestra temporale non contiene nodi utilizzabili")
    if coinbase_count and clusters < 2 and len(regular_indices):
        raise ValueError("con nodi coinbase e nodi normali servono almeno 2 cluster")

    # assegno -1 di default per i nodi coinbase e successivamente do una predizione solo ai nodi normali
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
    report_paths = split_report_paths("v1", args.graph_mode, "temporal.json")
    if args.output is not None:
        report_paths = {"analysis": args.output}
    args.graph_output = args.graph_output or analysis_artifact_path(
        "v1", args.graph_mode, "temporal_graph.png"
    )
    args.cluster_output = args.cluster_output or analysis_artifact_path(
        "v1", args.graph_mode, "temporal_clusters.png"
    )
    args.graph_mode = normalize_graph_mode(args.graph_mode)
    database = None  # Il loader legge DB_NAME e le credenziali dal .env.
    if (
        args.limit < 1
        or args.history_limit < 1
        or args.history_limit > args.limit
        or (args.clusters is not None and args.clusters < 1)
        or args.cluster_min < 1
        or args.cluster_max < args.cluster_min
    ):
        raise ValueError(
            "limit e clusters devono essere positivi e history-limit non può "
            "superare limit"
        )

    set_seed(args.seed)

    print("Caricamento storico e finestra di analisi dal database...", flush=True)

    wallet_manifest = None
    manifest_transaction_ids = None
    history_wallet_diagnostics = None
    analysis_wallet_diagnostics = None
    if args.graph_mode == "wallet":
        wallet_manifest = load_wallet_manifest()
        manifest_limit = int(wallet_manifest["limit"])
        manifest_history_limit = int(wallet_manifest["history_limit"])
        if (manifest_limit, manifest_history_limit) != (
            args.limit, args.history_limit
        ):
            raise ValueError(
                "il manifest wallet non corrisponde a --limit/--history-limit: "
                f"manifest={manifest_limit}/{manifest_history_limit}, "
                f"richiesto={args.limit}/{args.history_limit}"
            )
        manifest_transaction_ids = [
            int(value) for value in wallet_manifest["transaction_ids"]
        ]
        print("Uso i transaction_id del manifest wallet condiviso.")

    # Carico tutti i flussi senza le feature: il cutoff temporale deve essere
    # identico per storico e finestra, indipendentemente dal livello del grafo.
    all_flows, _ = load_v1_dataset(
        database,
        args.limit,
        history_limit=args.history_limit,
        include_features=False,
        transaction_ids=manifest_transaction_ids,
    )

    if wallet_manifest is not None:
        # Manteniamo anche gli ID che non producono flussi address -> address:
        # la divisione history/analysis deve restare quella del clustering.
        transaction_ids = manifest_transaction_ids
        history_ids = [int(value) for value in wallet_manifest["history_transaction_ids"]]
        analysis_ids = [int(value) for value in wallet_manifest["analysis_transaction_ids"]]
    else:
        transaction_ids = sorted(
            all_flows.transaction_id.dropna().astype(int).unique().tolist()
        )
        history_ids = transaction_ids[:args.history_limit]
        analysis_ids = transaction_ids[args.history_limit:args.limit]

    if not history_ids or not analysis_ids:
        raise ValueError("servono sia transazioni storiche sia transazioni nella finestra di analisi")

    # Separa fisicamente i flussi: il grafo storico non vede la finestra futura.
    # poi con il load_scoped_features carico le features di ogni nodo appartenente al flusso
    history_flows = all_flows[all_flows["transaction_id"].isin(history_ids)].copy()
    analysis_flows = all_flows[all_flows["transaction_id"].isin(analysis_ids)].copy()
    history_features = load_scoped_features(
        database, history_ids, flows=history_flows, split="history",
        include_wallet_features=args.graph_mode != "address",
    )
    analysis_features = load_scoped_features(
        database, analysis_ids, flows=analysis_flows, split="analysis",
        include_wallet_features=args.graph_mode != "address",
    )

    if args.graph_mode == "wallet":
        # Aggrego separatamente storico e finestra. In questo modo la GCN
        # storica non vede flussi futuri e la finestra mantiene il proprio
        # insieme di wallet e feature.
        history_flows, history_features = aggregate_wallet_graph_inputs(
            history_flows, history_features, split="history"
        )
        history_wallet_diagnostics = summarize_wallet_graph(
            history_flows, history_features
        )
        analysis_flows, analysis_features = aggregate_wallet_graph_inputs(
            analysis_flows, analysis_features, split="analysis"
        )
        analysis_wallet_diagnostics = summarize_wallet_graph(
            analysis_flows, analysis_features
        )

    # Graphs Data
    # costruisco i due grafi separati
    history_data = build_graph_data(
        history_flows, history_features, seed=args.seed,
        graph_mode=args.graph_mode,
        matrix_format=args.matrix_format,
    )
    analysis_data = build_graph_data(
        analysis_flows, analysis_features, seed=args.seed,
        graph_mode=args.graph_mode,
        matrix_format=args.matrix_format,
    )

    print(f"Transazioni storiche: {len(history_ids)}")
    print(f"Transazioni nella finestra: {len(analysis_ids)}")
    print(f"Livello grafo: {args.graph_mode}")
    print(f"Nodi storici: {len(history_data.node_names)}")
    print(f"Nodi nella finestra: {len(analysis_data.node_names)}")
    print("Nuovi indirizzi nella finestra: " f"{len(set(analysis_data.node_names) - set(history_data.node_names))}")
    print(f"Feature per nodo: {len(history_data.feature_names)}")
    print("\nTraining esclusivamente sullo storico...")

    # Il modello impara i pesi solo dal grafo storico.
    model = train_model(history_data, architecture=args.architecture)
    model.eval()

    # Faccio l'encode finale dello storico e della finestra, con i valori dei pesi aggiornati grazie al training effettuato sullo storico
    # Questo permette di calcolare embedding anche per indirizzi nuovi.
    with torch.no_grad():
        history_embeddings = model.encode(history_data.X, history_data.P)
        analysis_embeddings = model.encode(analysis_data.X, analysis_data.P)

    # trova indici storici che non sono coinbase
    history_regular_indices = torch.where(history_data.coinbase_nodes == False)[0]
    print(f"Embedding finestra: {tuple(analysis_embeddings.shape)}")

    # Il training GCN è già concluso. Ora proviamo più valori di k sugli stessi
    # embedding, imparando i centroidi soltanto dallo storico e assegnando la
    # finestra futura ai centroidi storici.
    candidate_clusters = (
        [args.clusters]
        if args.clusters is not None
        else list(range(args.cluster_min, args.cluster_max + 1))
    )
    sweep_results = []
    candidates = {}
    analysis_regular_indices = torch.where(analysis_data.coinbase_nodes == False)[0]
    for cluster_count in candidate_clusters:
        regular_clusters = (
            cluster_count - 1
            if int(analysis_data.coinbase_nodes.sum())
            else cluster_count
        )
        if regular_clusters < 1:
            raise ValueError("serve almeno un cluster per gli indirizzi non coinbase")
        if regular_clusters > len(history_regular_indices):
            raise ValueError("i cluster superano il numero di nodi non coinbase storici")

        _, historical_centroids = kmeans_fit(
            history_embeddings[history_regular_indices],
            n_clusters=regular_clusters,
        )
        candidate_assignments = _temporal_assignments(
            analysis_data, analysis_embeddings, cluster_count, historical_centroids
        )
        if len(analysis_regular_indices) > 0:
            candidate_silhouette = silhouette_score(
                analysis_embeddings[analysis_regular_indices],
                candidate_assignments[analysis_regular_indices],
            )
        else:
            candidate_silhouette = float("nan")
        candidate_roles = assign_roles(analysis_data, candidate_assignments)
        candidates[cluster_count] = (candidate_assignments, candidate_roles)
        sweep_results.append({
            "clusters": cluster_count,
            "silhouette_score": float(candidate_silhouette),
            "cluster_sizes": {
                str(cluster_id): int((candidate_assignments == cluster_id).sum())
                for cluster_id in sorted(set(candidate_assignments.tolist()))
            },
            "roles": {
                str(cluster_id): result["role"]
                for cluster_id, result in candidate_roles.items()
            },
        })
        print(
            f"k={cluster_count}: silhouette={candidate_silhouette:.4f} "
            f"cluster_sizes={sweep_results[-1]['cluster_sizes']}"
        )

    best_result = max(
        sweep_results,
        key=lambda result: result["silhouette_score"]
        if result["silhouette_score"] == result["silhouette_score"]
        else float("-inf"),
    )
    best_clusters = best_result["clusters"]
    assignments, roles = candidates[best_clusters]
    silhouette = best_result["silhouette_score"]
    print(f"Numero di cluster selezionato: {best_clusters}")

    # Stampo il grafo
    graph_output = plot_clustered_graph(
        analysis_data,
        assignments,
        args.graph_output,
        title="Role detection v1 temporale - grafo clusterizzato",
        seed=args.seed,
    )
    print(f"Grafo clusterizzato salvato in: {graph_output}")
    cluster_output = plot_embedding_clusters(
        analysis_embeddings,
        assignments,
        args.cluster_output,
        title="Role detection v1 temporale - cluster degli embedding",
    )
    print(f"Visualizzazione cluster salvata in: {cluster_output}")
    print(f"Silhouette score finestra: {silhouette:.4f}")
    describe_v1_clusters(analysis_data, assignments)
    roles = assign_roles(analysis_data, assignments)

    print("\nRuoli assegnati nella finestra:")
    for cluster_id in sorted(roles):
        result = roles[cluster_id]
        print(f"  Cluster {cluster_id}: {result['role']} ({result['size']} indirizzi)")
        print(f"    Wallet rappresentativo: {result['representative']}")

    report = {
        "experiment": {
            "database": "from_env",
            "mode": "temporal",
            "history_transaction_ids": history_ids,
            "analysis_transaction_ids": analysis_ids,
            "history_limit": args.history_limit,
            "analysis_limit": len(analysis_ids),
            "seed": args.seed,
            "architecture": args.architecture,
            "matrix_format": args.matrix_format,
            "graph_mode": args.graph_mode,
            "clusters": best_clusters,
            "cluster_sweep": sweep_results,
            "history_num_nodes": len(history_data.node_names),
            "analysis_num_nodes": len(analysis_data.node_names),
            "new_analysis_nodes": len(
                set(analysis_data.node_names) - set(history_data.node_names)
            ),
            "analysis_num_edges": int(analysis_data.positive_edges.shape[1]),
            "analysis_coinbase_nodes": int(analysis_data.coinbase_nodes.sum()),
            "embedding_dim": int(analysis_embeddings.shape[1]),
            "silhouette_score": silhouette,
            "history_wallet_graph": history_wallet_diagnostics,
            "analysis_wallet_graph": analysis_wallet_diagnostics,
        },
        "analysis_nodes": analysis_data.node_names,
        "analysis_embeddings": analysis_embeddings.cpu().tolist(),
        "assignments": assignments.cpu().tolist(),
        "roles": {str(cluster_id): result for cluster_id, result in roles.items()},
    }
    write_split_json(report, report_paths)
    print("\nReport salvati in:")
    for path in report_paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
