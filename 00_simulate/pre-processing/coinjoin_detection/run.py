#!/usr/bin/env python3
"""Calcola e salva nel database lo score CoinJoin delle transazioni."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold deve essere compreso tra 0 e 1")

    try:
        import psycopg2

        with psycopg2.connect(**database_parameters(args.database)) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    ALTER TABLE transaction
                    ADD COLUMN IF NOT EXISTS coinjoin_score NUMERIC(6,5)
                """)
                cursor.execute("""
                    SELECT t.id, t.n_inputs,
                           COALESCE(
                               ARRAY_AGG(o.amount ORDER BY o.n)
                                   FILTER (WHERE o.id IS NOT NULL),
                               ARRAY[]::numeric[]
                           ) AS output_amounts
                    FROM transaction t
                    LEFT JOIN tx_output o ON o.fk_transaction_id = t.id
                    WHERE t.n_inputs > 0
                    GROUP BY t.id, t.n_inputs
                    ORDER BY t.id
                """)
                rows = cursor.fetchall()
                updates = []
                for transaction_id, n_inputs, output_amounts in rows:
                    result = classify_transaction(
                        n_inputs, output_amounts
                    )
                    updates.append(
                        (
                            result["score"],
                            transaction_id,
                        )
                    )
                cursor.executemany("""
                    UPDATE transaction
                    SET coinjoin_score = %s
                    WHERE id = %s
                """, updates)
                cursor.execute("""
                    UPDATE transaction
                    SET coinjoin_score = 0
                    WHERE n_inputs = 0
                """)
                candidate_count = sum(
                    1
                    for transaction_id, n_inputs, output_amounts in rows
                    if classify_transaction(
                        n_inputs, output_amounts
                    )["score"] >= args.threshold
                )
                print(
                    f"Transazioni analizzate: {len(rows)} | "
                    f"CoinJoin candidate: {candidate_count}"
                )
    except Exception as exc:
        print(f"ERRORE: impossibile calcolare lo score CoinJoin: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
