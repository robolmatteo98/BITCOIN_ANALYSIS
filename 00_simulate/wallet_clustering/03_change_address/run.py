"""Esegue il detector preliminare dei change address su dati statici."""

from __future__ import annotations

from algorithm import find_change_candidates
from example_data import example_inputs, example_outputs


def main() -> int:
    candidates = find_change_candidates(
        example_inputs(),
        example_outputs(),
        threshold=0.75,
    )
    print("Candidati change address:")
    for row in candidates.itertuples(index=False):
        if row.is_change_candidate:
            print(
                f"  {row.tx_id} output {row.output_index}: {row.address} "
                f"(score={row.score:.2f}; {', '.join(row.reasons)})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
