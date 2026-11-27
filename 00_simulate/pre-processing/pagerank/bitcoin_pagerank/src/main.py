"""Entry point for the database-backed PageRank preprocessing job."""

import argparse
import sys
from pathlib import Path

from pyspark import SparkConf, SparkContext

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bitcoin_common.bootstrap import configure_imports
from bitcoin_common.pagerank_runtime import add_python_files_to_workers
from utils.aggregate_edges import aggregate_edges_by_pair
from utils.load_db import load_db_edges
from utils.mode.paper.pagerank import run_pagerank as run_paper_pagerank
from utils.mode.simple.pagerank import run_pagerank as run_simple_pagerank
from utils.print_output import write_output

configure_imports(__file__)

DEFAULT_DAMPING = 0.85
DEFAULT_OUTPUT_DIR = Path("07_reports/pre-processing/pagerank")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcola PageRank sugli archi letti da PostgreSQL."
    )
    parser.add_argument("--database", default=None, help="Nome database PostgreSQL.")
    parser.add_argument(
        "--wallet-mapping-dir", default=None,
        help="Cartella wallet_clustering con le sottocartelle history/analysis.",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Cartella radice dei risultati history/analysis.",
    )
    parser.add_argument("--mode", choices=["basic", "combined", "paper"], default="basic")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--limit", type=int, default=11000,
        help="Cutoff totale delle transazioni (default: 11000).",
    )
    parser.add_argument(
        "--history-limit", type=int, default=10000,
        help="Numero di transazioni storiche (default: 10000).",
    )
    return parser.parse_args()


def run_split(sc, args, split, transaction_limit, transaction_offset, wallet_mapping):
    """Esegue una partizione logica e salva i relativi risultati."""
    raw_edges, db_stats = load_db_edges(
        database=args.database,
        transaction_limit=transaction_limit,
        transaction_offset=transaction_offset,
        wallet_mapping=wallet_mapping,
    )
    raw_edges_rdd = sc.parallelize(raw_edges)
    timestamps = [edge["timestamp"] for edge in raw_edges if edge["timestamp"] is not None]
    time_bounds = (min(timestamps), max(timestamps)) if timestamps else (None, None)
    edges_rdd = aggregate_edges_by_pair(raw_edges_rdd).cache()

    if args.mode == "paper":
        ranks = run_paper_pagerank(
            edges_rdd, args.iterations, DEFAULT_DAMPING, time_bounds
        ).cache()
    else:
        ranks = run_simple_pagerank(
            edges_rdd, args.mode, args.iterations, DEFAULT_DAMPING, time_bounds
        ).cache()

    top = ranks.takeOrdered(20, key=lambda item: -item[1])
    write_output(
        args=args,
        top=top,
        ranks=ranks,
        raw_edge_count=len(raw_edges),
        edges_rdd=edges_rdd,
        output_dir=Path(args.output_dir) / split,
        split=split,
        transaction_limit=transaction_limit,
        transaction_offset=transaction_offset,
        wallet_mapping=wallet_mapping,
        db_stats=db_stats,
    )
    edges_rdd.unpersist()
    ranks.unpersist()


def main():
    args = parse_args()
    conf = SparkConf().setAppName(f"BitcoinPageRank-{args.mode}")
    sc = SparkContext(conf=conf)
    add_python_files_to_workers(sc)

    mapping_dir = Path(args.wallet_mapping_dir) if args.wallet_mapping_dir else None
    if args.history_limit < 1 or args.history_limit >= args.limit:
        raise ValueError("history-limit deve essere positivo e minore di limit")

    history_limit = args.history_limit
    analysis_limit = args.limit - args.history_limit

    try:
        run_split(
            sc, args, "history", history_limit, 0,
            mapping_dir / "history" / "address_to_wallet.csv" if mapping_dir else None,
        )
        run_split(
            sc, args, "analysis", analysis_limit, history_limit,
            mapping_dir / "analysis" / "address_to_wallet.csv" if mapping_dir else None,
        )
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
