"""Esegue il co-spending su un piccolo db statico."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithm import find_clusters
from example_data import example_inputs


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "07_reports" / "wallet_clustering" / "co-spending",
        help="cartella in cui salvare i cluster",
    )
    args = parser.parse_args()

    # Carico i dati
    inputs = example_inputs()

    # Creo i clusters
    clusters = find_clusters(inputs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_rows = [
        {"wallet_id": index, "addresses": sorted(cluster)}
        for index, cluster in enumerate(clusters, start=1)
    ]
    address_to_wallet = {
        address: row["wallet_id"]
        for row in cluster_rows
        for address in row["addresses"]
    }
    output = {
        "clusters": cluster_rows,
        "address_to_wallet": address_to_wallet,
    }
    output_path = args.output_dir / "co_spending_clusters.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Transazioni: {inputs.tx_id.nunique() if not inputs.empty else 0}")
    print(f"Indirizzi sorgente: {inputs.source.nunique() if not inputs.empty else 0}")
    print(f"Cluster: {len(clusters)}")
    for cluster_id, cluster in enumerate(clusters, start=1):
        print(f"Cluster {cluster_id} ({len(cluster)} indirizzi): {', '.join(sorted(cluster))}")
    print(f"Risultati salvati in: {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
