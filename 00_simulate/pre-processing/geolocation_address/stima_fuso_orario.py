"""Stima la regione geografica degli address tramite gli orari di attività."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from collections import Counter
from pathlib import Path

import psycopg2
from dotenv import find_dotenv, load_dotenv


ROOT = Path(__file__).resolve().parents[3]

# Intervalli geografici usati per interpretare l'offset stimato.
REGIONS = [
    ("America del Nord (West)", -12, -8),
    ("America del Nord / Sud", -7, -3),
    ("Europa Occidentale / Africa Occidentale", -2, 0),
    ("Europa Centrale / Africa", 1, 3),
    ("Asia Occidentale / India", 4, 6),
    ("Asia Orientale", 7, 9),
    ("Oceania / Asia Pacifico", 10, 12),
]


def database_parameters(database: str | None) -> dict[str, object]:
    """Legge i parametri PostgreSQL dal file .env del progetto."""
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


def guess_region_name(utc_offset: int) -> str | None:
    for name, start, end in REGIONS:
        if start <= utc_offset <= end:
            return name
    return None


def load_addresses(cursor, transaction_ids: list[int]) -> list[str]:
    """Recupera gli address presenti negli input e negli output selezionati."""
    cursor.execute(
        """
        WITH selected_addresses AS (
            SELECT o.fk_address_id AS address_id
            FROM tx_output o
            WHERE o.fk_transaction_id = ANY(%s)
              AND o.fk_address_id IS NOT NULL
            UNION
            SELECT previous_output.fk_address_id AS address_id
            FROM tx_input i
            JOIN tx_output previous_output
              ON previous_output.fk_transaction_id = i.prev_transaction_id
             AND previous_output.n = i.prev_vout
            WHERE i.fk_transaction_id = ANY(%s)
              AND previous_output.fk_address_id IS NOT NULL
        )
        SELECT DISTINCT a.code
        FROM selected_addresses selected
        JOIN address a ON a.id = selected.address_id
        ORDER BY a.code
        """,
        (transaction_ids, transaction_ids),
    )
    return [row[0] for row in cursor.fetchall()]


def estimate_address(cursor, address: str, transaction_ids: list[int]) -> dict[str, object]:
    """Calcola la stima geografica di un address su una partizione."""
    cursor.execute(
        """
        SELECT b.time
        FROM tx_output o
        JOIN transaction t ON t.id = o.fk_transaction_id
        JOIN block b ON b.id = t.fk_block_id
        JOIN address a ON a.id = o.fk_address_id
        WHERE o.fk_transaction_id = ANY(%s)
          AND a.code = %s
        UNION ALL
        SELECT b.time
        FROM tx_input i
        JOIN tx_output previous_output
          ON previous_output.fk_transaction_id = i.prev_transaction_id
         AND previous_output.n = i.prev_vout
        JOIN transaction t ON t.id = i.fk_transaction_id
        JOIN block b ON b.id = t.fk_block_id
        JOIN address a ON a.id = previous_output.fk_address_id
        WHERE i.fk_transaction_id = ANY(%s)
          AND a.code = %s
        """,
        (transaction_ids, address, transaction_ids, address),
    )
    timestamps = [row[0] for row in cursor.fetchall()]

    # Ore UTC delle transazioni.
    hours = [datetime.datetime.utcfromtimestamp(ts).hour for ts in timestamps]
    hour_counts = Counter(hours)
    active_hours = [
        hour
        for hour, count in sorted(hour_counts.items())
        for _ in range(count)
    ]

    # Testiamo tutti i fusi orari (offset da -12 a +12).
    best_offset = 0
    best_activity_score = 0
    active_range = list(range(10, 23))  # ore diurne 10-22
    for offset in range(-12, 13):
        shifted_hours = [(hour + offset) % 24 for hour in active_hours]
        score = sum(1 for hour in shifted_hours if hour in active_range)
        if score > best_activity_score:
            best_activity_score = score
            best_offset = offset

    return {
        "address": address,
        "transaction_count": len(timestamps),
        "estimated_utc_offset": best_offset,
        "activity_score": best_activity_score,
        "region": guess_region_name(best_offset),
    }


def write_results(path: Path, results: list[dict[str, object]]) -> None:
    fieldnames = [
        "address", "split", "transaction_count", "estimated_utc_offset",
        "activity_score", "region",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", metavar="NOME_DB")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--history-limit", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "07_reports" / "pre-processing" / "geolocation",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit deve essere positivo")
    if args.history_limit < 0:
        parser.error("--history-limit non può essere negativo")
    if args.limit is not None and args.history_limit > args.limit:
        parser.error("--history-limit non può superare --limit")

    limit_clause = "LIMIT %s" if args.limit is not None else ""
    try:
        with psycopg2.connect(**database_parameters(args.database)) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id FROM transaction ORDER BY id {limit_clause}",
                    (args.limit,) if args.limit is not None else (),
                )
                transaction_ids = [row[0] for row in cursor.fetchall()]
                history_ids = transaction_ids[:args.history_limit]
                analysis_ids = (
                    transaction_ids[args.history_limit:]
                    if args.history_limit
                    else transaction_ids
                )

                results = []
                for split, split_ids in (
                    ("history", history_ids),
                    ("analysis", analysis_ids),
                ):
                    addresses = load_addresses(cursor, split_ids) if split_ids else []
                    for index, address in enumerate(addresses, start=1):
                        result = estimate_address(cursor, address, split_ids)
                        result["split"] = split
                        results.append(result)
                        print(
                            f"{split}: address geolocalizzati "
                            f"{index} / {len(addresses)}",
                            end="\r",
                        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "address_geolocation.csv"
        write_results(output_path, results)

        for split in ("history", "analysis"):
            split_dir = args.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            write_results(
                split_dir / "address_geolocation.csv",
                [row for row in results if row["split"] == split],
            )

        summary = {
            "limit": args.limit,
            "history_limit": args.history_limit,
            "total_transaction_count": len(transaction_ids),
            "history_transaction_count": len(history_ids),
            "analysis_transaction_count": len(analysis_ids),
            "address_count": len(results),
            "output_file": str(output_path),
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Address geolocalizzati: {len(results)}")
        print(f"Risultati salvati in: {output_path}")
        print(f"Riepilogo salvato in: {summary_path}")
    except Exception as exc:
        print(f"ERRORE: impossibile calcolare la geolocation: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
