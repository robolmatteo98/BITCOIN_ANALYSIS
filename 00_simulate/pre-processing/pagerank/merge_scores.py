"""Unisce i PageRank delle tre modalità in una tabella unica per split."""

import csv
import argparse
from pathlib import Path


MODES = ("basic", "combined", "paper")


def read_scores(path: Path) -> dict[str, float]:
    """Legge un file PageRank e restituisce node -> score."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["node"]: float(row["score"])
            for row in csv.DictReader(handle)
            if row.get("node")
        }


def merge_split(split_dir: Path) -> Path:
    scores = {
        mode: read_scores(split_dir / f"pagerank_{mode}_all.csv")
        for mode in MODES
    }
    nodes = sorted(set().union(*(values.keys() for values in scores.values())))
    output_path = split_dir / "pagerank_scores.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", *(f"pagerank_{mode}" for mode in MODES)])
        for node in nodes:
            writer.writerow([
                node,
                *(scores[mode].get(node, 0.0) for mode in MODES),
            ])
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="07_reports/pre-processing/pagerank",
        help="Cartella che contiene history e analysis.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    for split in ("history", "analysis"):
        output_path = merge_split(root / split)
        print(f"Creato: {output_path}")


if __name__ == "__main__":
    main()
