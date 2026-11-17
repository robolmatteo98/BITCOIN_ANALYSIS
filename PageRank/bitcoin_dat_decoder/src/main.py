import argparse
import os
import sys
from pathlib import Path

from pyspark import SparkContext, StorageLevel
from pyspark import SparkConf
from pyspark.sql import SparkSession

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bitcoin_common.bootstrap import configure_imports
from bitcoin_common.decoder_runtime import add_python_files_to_workers
from bitcoin_common.shared_bitcoin_graph import block_to_edge_rows
from utils.blocks_decoder import parse_local_dat_file
from utils.common import clean_output_dir, ensure_dir
from utils.print_output import print_summary
from utils.load_dat import select_dat_files_from_glob
from utils.normalize import build_edges_df

configure_imports(__file__)

def main():
    parser = argparse.ArgumentParser(description="Decoder Bitcoin .dat")
    parser.add_argument("--input-glob", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sample-files", type=int, default=None)
    args = parser.parse_args()

    if not args.input_glob or not args.output_root:
        raise ValueError("In modalita spark servono --input-glob e --output-root")
    
    selected_files = select_dat_files_from_glob(args.input_glob, args.sample_files)
    if not selected_files:
        raise FileNotFoundError(f"Nessun file .dat trovato per il pattern: {args.input_glob}")

    output_root = ensure_dir(args.output_root)
    csv_dir = output_root / "edges_csv"
    output_path = output_root / "summary.txt"

    print("File .dat selezionati:")
    for path in selected_files:
        print(str(path))

    # Creazione contesto SPARK
    conf = SparkConf().setAppName("BitcoinDatIngestion")
    
    sc = SparkContext(conf=conf)
    add_python_files_to_workers(sc)

    spark = SparkSession.builder.getOrCreate()
    selected_paths = [str(path) for path in selected_files]

    # STEP 1: distribuisce i percorsi dei file creando una partizione per ogni singolo file .dat
    dat_paths_rdd = sc.parallelize(selected_paths, len(selected_paths))

    # STEP 2: ogni task apre, legge e parsa in modo sequenziale il file .dat
    blocks_rdd = dat_paths_rdd.flatMap(parse_local_dat_file)
    block_count = blocks_rdd.count()
    print(f"Numero blocchi estratti: {block_count}")

    # STEP 3: dai blocchi vengono costruiti edge list persistenti per i job successivi.
    edges_rdd = blocks_rdd.flatMap(block_to_edge_rows).persist(StorageLevel.DISK_ONLY)
    edge_count = edges_rdd.count()
    print(f"Numero archi estratti: {edge_count}")

    # STEP 4: edge list aggiornata, serve a trasformare RDD di dizionari in DataFrame Spark con schema esplicito
    edges_df = build_edges_df(spark, edges_rdd)
    
    # STEP 5: cancella eventuali file contenuti nella cartella dove verrò salvato il .csv
    clean_output_dir(csv_dir)

    # STEP 6: stampa i risultati a video e su disco
    edges_df.write.mode("overwrite").option("header", True).csv(str(csv_dir))
    print_summary(selected_paths, block_count, edge_count, output_path)

    sc.stop()

if __name__ == "__main__":
    main()