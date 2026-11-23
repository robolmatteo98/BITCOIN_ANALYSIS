"""Esegue il detector preliminare dei change address su dati statici."""

from __future__ import annotations

import argparse
from pathlib import Path

from algorithm import find_change_candidates
from example_data import example_inputs, example_outputs


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "07_reports" / "wallet_clustering" / "change_address",
        help="cartella in cui salvare i candidati",
    )
    args = parser.parse_args()

    candidates = find_change_candidates(
        example_inputs(),
        example_outputs(),
        threshold=0.75,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = candidates.copy()
    output["reasons"] = output["reasons"].apply("; ".join)
    output_path = args.output_dir / "change_candidates.csv"
    output.to_csv(output_path, index=False)
    print("Candidati change address:")
    for row in candidates.itertuples(index=False):
        if row.is_change_candidate:
            print(
                f"  {row.tx_id} output {row.output_index}: {row.address} "
                f"(score={row.score:.2f}; {', '.join(row.reasons)})"
            )
    print(f"Risultati salvati in: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
