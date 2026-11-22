"""Caricamento dal database dei dati necessari alla role detection v1."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

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


def load_v1_dataset(
    database: str | None = None,
    limit: int = 1000,
    history_limit: int = 0,
    include_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica contesto storico e finestra di analisi fino allo stesso cutoff."""
    if limit < 1 or history_limit < 0:
        raise ValueError("limit deve essere positivo e history_limit non negativo")

    total_limit = history_limit + limit

    # Seleziono contesto + finestra in ordine di transaction.id. La selezione
    # include anche le coinbase, purché abbiano almeno un output decodificato.
    # Nessun dato successivo al cutoff entra nel DB.
    flows_query = """
        WITH selected_transactions AS (
            SELECT DISTINCT t.id AS transaction_id
            FROM transaction t
            JOIN tx_output output
              ON output.fk_transaction_id = t.id
            JOIN address output_address
              ON output_address.id = output.fk_address_id
            WHERE t.n_inputs = 0
               OR EXISTS (
                    SELECT 1
                    FROM tx_input input_row
                    JOIN tx_output previous_output
                      ON previous_output.fk_transaction_id = input_row.prev_transaction_id
                     AND previous_output.n = input_row.prev_vout
                    JOIN address input_address
                      ON input_address.id = previous_output.fk_address_id
                    WHERE input_row.fk_transaction_id = t.id
               )
            ORDER BY t.id
            LIMIT %s
        )
        SELECT f.from_address, f.to_address, f.flow_amount, f.time,
               t.id AS transaction_id,
               (t.n_inputs = 0) AS is_coinbase
        FROM flows_view f
        JOIN transaction t
          ON t.id = f.txid
        JOIN selected_transactions s
          ON t.id = s.transaction_id
        WHERE f.to_address IS NOT NULL
        ORDER BY t.id, f.from_address, f.to_address
    """

    with _database_connection(database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(flows_query, (total_limit,))
            flow_values = cursor.fetchall()
            flow_columns = [description[0] for description in cursor.description]

        flows = pd.DataFrame(flow_values, columns=flow_columns)
        if flows.empty:
            return flows, pd.DataFrame()
        if not include_features:
            return flows, pd.DataFrame()

        # Le feature vengono calcolate dalla funzione SQL sullo stesso insieme di txid, evitando di leggere la address_feature_summary globale.
        tx_ids = flows["transaction_id"].dropna().astype(int).unique().tolist()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM address_feature_summary_for_transactions(%s::bigint[])
                """,
                (tx_ids,),
            )
            feature_values = cursor.fetchall()
            feature_columns = [description[0] for description in cursor.description]

    # flows = una riga per ogni flusso address → address
    # features = una riga per ogni address
    return flows, pd.DataFrame(feature_values, columns=feature_columns)


def load_scoped_features(
    database: str | None,
    transaction_ids: list[int],
) -> pd.DataFrame:
    """Carica le feature SQL limitandole a un insieme di transazioni."""
    if not transaction_ids:
        return pd.DataFrame()

    with _database_connection(database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM address_feature_summary_for_transactions(%s::bigint[])
                """,
                (transaction_ids,),
            )
            feature_values = cursor.fetchall()
            feature_columns = [description[0] for description in cursor.description]

    return pd.DataFrame(feature_values, columns=feature_columns)
