"""Interrogazione PostgreSQL per la role detection."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

from .graph_builder import GraphData, build_graph_data


PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
            SELECT DISTINCT txid
            FROM flows_view
            WHERE from_address IS NOT NULL
              AND to_address IS NOT NULL
            ORDER BY txid
            LIMIT %s
        )
        SELECT f.from_address, f.to_address, f.flow_amount, f.time, f.txid
        FROM flows_view f
        JOIN selected_transactions s USING (txid)
        WHERE f.from_address IS NOT NULL
          AND f.to_address IS NOT NULL
        ORDER BY f.txid, f.from_address, f.to_address
    """

    with _database_connection(database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (limit,))
            values = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

    return pd.DataFrame(values, columns=columns)


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
