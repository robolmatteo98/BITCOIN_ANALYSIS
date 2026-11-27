"""Interrogazione PostgreSQL per la role detection."""

from __future__ import annotations

import os
import csv
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

from .graph_builder import GraphData, build_graph_data


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WALLET_REPORTS_ROOT = PROJECT_ROOT / "07_reports" / "wallet_clustering" / "co-spending"


def _database_connection(database: str | None = None):
    load_dotenv(find_dotenv() or PROJECT_ROOT / ".env")
    required = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"variabili mancanti nel .env: {', '.join(missing)}")

    selected_database = database or os.getenv("DB_NAME")
    if not selected_database:
        raise RuntimeError("specificare database oppure configurare DB_NAME")

    return psycopg2.connect(
        database=selected_database,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )


def load_transactions(database: str | None = None, limit: int = 1000) -> pd.DataFrame:
    """Carica le prime ``limit`` transazioni con flussi address → address."""
    if limit < 1:
        raise ValueError("limit deve essere positivo")

    query = """
        WITH selected_transactions AS (
            SELECT DISTINCT t.id AS transaction_id
            FROM transaction t
            JOIN flows_view f
              ON f.txid = t.id
            WHERE f.from_address IS NOT NULL
              AND f.to_address IS NOT NULL
            ORDER BY t.id
            LIMIT %s
        )
        SELECT f.from_address, f.to_address, f.flow_amount, f.time,
               t.id AS transaction_id, t.txid
        FROM flows_view f
        JOIN transaction t
          ON t.id = f.txid
        JOIN selected_transactions s
          ON s.transaction_id = t.id
        WHERE f.from_address IS NOT NULL
          AND f.to_address IS NOT NULL
        ORDER BY t.id, f.from_address, f.to_address
    """

    with _database_connection(database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (limit,))
            values = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

    return pd.DataFrame(values, columns=columns)


def _load_wallet_mapping(split: str) -> dict[str, int]:
    """Carica il mapping address -> wallet della partizione richiesta."""
    path = WALLET_REPORTS_ROOT / split / "address_to_wallet.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"manca il mapping wallet per la partizione {split}: {path}"
        )
    with path.open(newline="", encoding="utf-8") as mapping_file:
        return {
            row["address"]: int(row["wallet_id"])
            for row in csv.DictReader(mapping_file)
        }


def load_combined_wallet_mapping() -> dict[str, int]:
    """Unisce mapping storico e analisi mantenendo gli stessi wallet_id."""
    mapping = _load_wallet_mapping("history")
    mapping.update(_load_wallet_mapping("analysis"))
    return mapping


def complete_wallet_mapping(
    rows: pd.DataFrame,
    mapping: dict[str, int],
) -> dict[str, int]:
    """Aggiunge wallet singleton per gli endpoint senza clustering."""
    completed = mapping.copy()
    next_wallet_id = max(completed.values(), default=0) + 1
    addresses = set(rows["from_address"].dropna().astype(str))
    addresses.update(rows["to_address"].dropna().astype(str))
    for address in sorted(addresses - set(completed)):
        completed[address] = next_wallet_id
        next_wallet_id += 1
    return completed


def aggregate_wallet_transactions(
    rows: pd.DataFrame,
    mapping: dict[str, int],
) -> pd.DataFrame:
    """Sostituisce gli address con wallet_id e scarta flussi non mappati."""
    aggregated = rows.copy()
    aggregated["from_address"] = aggregated["from_address"].map(mapping)
    aggregated["to_address"] = aggregated["to_address"].map(mapping)
    valid = aggregated["from_address"].notna() & aggregated["to_address"].notna()
    aggregated = aggregated[valid].copy()
    aggregated["from_address"] = aggregated["from_address"].map(
        lambda wallet_id: f"wallet_{int(wallet_id)}"
    )
    aggregated["to_address"] = aggregated["to_address"].map(
        lambda wallet_id: f"wallet_{int(wallet_id)}"
    )
    return aggregated


def prepare_graph_from_db(
    database: str | None = None,
    limit: int = 1000,
    seed: int = 42,
) -> GraphData:
    """Legge il DB e usa il costruttore comune del grafo."""
    rows = load_transactions(database=database, limit=limit)
    transactions = [
        (row.from_address, row.to_address, row.flow_amount, row.time)
        for row in rows.itertuples(index=False)
    ]
    return build_graph_data(transactions, seed=seed)
