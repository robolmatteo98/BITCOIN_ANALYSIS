import argparse
import os
import sys
from pathlib import Path

from pyspark import SparkConf, SparkContext

# configurazione moduli python per docker o azure.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bitcoin_common.bootstrap import configure_imports
from bitcoin_common.pagerank_runtime import add_python_files_to_workers
from utils.aggregate_edges import aggregate_edges_by_pair
from utils.print_output import write_output
from utils.load_blocks import load_edges_csv_rdd, compute_time_bounds
from utils.mode.simple.pagerank import run_pagerank as run_simple_pagerank
from utils.mode.paper.pagerank import run_pagerank as run_paper_pagerank

configure_imports(__file__)

DEFAULT_DAMPING = 0.85

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["basic", "combined", "paper"], default="basic")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    # Configurazione ambiente Spark
    conf = SparkConf().setAppName(f"BitcoinLabStyle-{args.mode}")
    
    sc = SparkContext(conf=conf)
    add_python_files_to_workers(sc)

    # STEP 1: carico i csv in un rdd
    raw_edges_rdd = load_edges_csv_rdd(sc, args.input_glob)
    raw_edge_count = raw_edges_rdd.count()

    # STEP 1.1: calcolo una sola volta i bounds temporali del dataset.
    time_bounds = compute_time_bounds(raw_edges_rdd)

    # STEP 2: semplifico il numero di archi tramite l'aggregazione 
    edges_rdd = aggregate_edges_by_pair(raw_edges_rdd).cache()

    # STEP 3: decido quale modalità lanciare
    if args.mode == "paper":
        ranks = run_paper_pagerank(edges_rdd, args.iterations, DEFAULT_DAMPING, time_bounds).cache()
    else:
        ranks = run_simple_pagerank(edges_rdd, args.mode, args.iterations, DEFAULT_DAMPING, time_bounds).cache()

    # STEP 4: stampo i risultati a video e su disco
    top = ranks.takeOrdered(20, key=lambda item: -item[1])

    write_output(args, top, raw_edge_count, edges_rdd)

    sc.stop()

if __name__ == "__main__":
    main()