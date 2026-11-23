#!/usr/bin/env python3
"""Calcola e salva su file lo score CoinJoin delle transazioni."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithm import classify_transaction  # noqa: E402


def database_parameters(database: str | None) -> dict[str, object]:
    load_dotenv(find_dotenv() or ROOT / ".env")
    selected_database = database or os.getenv("DB_NAME")
    required = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"variabili mancanti nel .env: {', '.join(missing)}")
    if not selected_database:
        raise RuntimeError("specificare --database oppure configurare DB_NAME")
    return {
        "dbname": selected_database,
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", metavar="NOME_DB")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "07_reports" / "pre-processing" / "coinjoin",
        help="cartella in cui salvare score e riepilogo",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold deve essere compreso tra 0 e 1")

    try:
        import psycopg2

        with psycopg2.connect(**database_parameters(args.database)) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT t.id, t.n_inputs,
                           COALESCE(
                               ARRAY_AGG(o.amount ORDER BY o.n)
                                   FILTER (WHERE o.id IS NOT NULL),
                               ARRAY[]::numeric[]
                           ) AS output_amounts
                    FROM transaction t
                    LEFT JOIN tx_output o ON o.fk_transaction_id = t.id
                    GROUP BY t.id, t.n_inputs
                    ORDER BY t.id
                """)
                rows = cursor.fetchall()
                results = []
                for transaction_id, n_inputs, output_amounts in rows:
                    # Calcolo lo score senza modificare il database.
                    result = (
                        {"score": 0.0, "signals": {}, "n_inputs": n_inputs,
                         "n_outputs": len(output_amounts or [])}
                        if n_inputs == 0
                        else classify_transaction(n_inputs, output_amounts)
                    )
                    signals = result["signals"]
                    results.append(
                        {
                            "transaction_id": transaction_id,
                            "n_inputs": n_inputs,
                            "n_outputs": result["n_outputs"],
                            "coinjoin_score": result["score"],
                            "candidate": result["score"] >= args.threshold,
                            "has_multiple_inputs_outputs": signals.get(
                                "has_multiple_inputs_outputs", False
                            ),
                            "has_participant_shape": signals.get(
                                "has_participant_shape", False
                            ),
                            "has_repeated_output_amount": signals.get(
                                "has_repeated_output_amount", False
                            ),
                            "repeated_output_share": signals.get(
                                "repeated_output_share", 0.0
                            ),
                            "repeated_amount_groups": signals.get(
                                "repeated_amount_groups", 0
                            ),
                            "largest_equal_group": signals.get(
                                "largest_equal_group", 0
                            ),
                        }
                    )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "coinjoin_scores.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as output_file:
            fieldnames = list(results[0]) if results else [
                "transaction_id", "n_inputs", "n_outputs", "coinjoin_score",
                "candidate",
            ]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        candidate_count = sum(row["candidate"] for row in results)
        summary = {
            "threshold": args.threshold,
            "transaction_count": len(results),
            "candidate_count": candidate_count,
            "output_file": str(csv_path),
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Transazioni analizzate: {len(results)} | "
            f"CoinJoin candidate: {candidate_count}"
        )
        print(f"Risultati salvati in: {csv_path}")
        print(f"Riepilogo salvato in: {summary_path}")
    except Exception as exc:
        print(f"ERRORE: impossibile calcolare lo score CoinJoin: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
