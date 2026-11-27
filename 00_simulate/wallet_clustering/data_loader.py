"""Caricamento dei dati Bitcoin necessari alla pipeline di clustering."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from db import database_parameters


WALLET_ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    """Carica un detector dal suo file senza ambiguita tra gli algorithm.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"impossibile caricare {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_transactions(database: str | None, limit: int | None):
    """Legge transazioni, input, output, script type e score CoinJoin.

    Gli input ricevono lo script type dell'output precedente che stanno
    spendendo. Questo e il tipo corretto da confrontare con gli output nuovi.
    """
    import psycopg2

    limit_clause = "LIMIT %s" if limit is not None else ""
    with psycopg2.connect(**database_parameters(database)) as connection:
        with connection.cursor() as cursor:
            # Il prefisso ordinato verra usato come storico; il resto sara la
            # finestra futura di analisi.
            cursor.execute(
                f"SELECT id FROM transaction ORDER BY id {limit_clause}",
                (limit,) if limit is not None else (),
            )
            transaction_ids = [row[0] for row in cursor.fetchall()]
            if not transaction_ids:
                return (
                    pd.DataFrame(columns=["tx_id", "address", "amount", "timestamp", "script_type"]),
                    pd.DataFrame(columns=["tx_id", "output_index", "address", "amount", "script_type"]),
                    pd.DataFrame(columns=["transaction_id", "coinjoin_score", "candidate"]),
                    transaction_ids,
                )

            # Un input non contiene il proprio script di pagamento: lo
            # recuperiamo dall'output precedente tramite prev_transaction_id e
            # prev_vout.
            cursor.execute(
                """
                SELECT i.fk_transaction_id AS tx_id, a.code AS address,
                       previous_output.amount, b.time AS timestamp,
                       COALESCE(st.code, 'UNKNOWN') AS script_type
                FROM tx_input i
                JOIN tx_output previous_output
                  ON previous_output.fk_transaction_id = i.prev_transaction_id
                 AND previous_output.n = i.prev_vout
                JOIN address a ON a.id = previous_output.fk_address_id
                LEFT JOIN script_type st ON st.id = previous_output.fk_script_type_id
                JOIN transaction t ON t.id = i.fk_transaction_id
                JOIN block b ON b.id = t.fk_block_id
                WHERE i.fk_transaction_id = ANY(%s) AND a.code IS NOT NULL
                ORDER BY i.fk_transaction_id, i.id
                """,
                (transaction_ids,),
            )
            inputs = pd.DataFrame(
                cursor.fetchall(),
                columns=["tx_id", "address", "amount", "timestamp", "script_type"],
            )

            # Gli output correnti sono i candidati change e portano il tipo di
            # script assegnato dal decoder dei file .dat.
            cursor.execute(
                """
                SELECT o.fk_transaction_id AS tx_id, o.n AS output_index,
                       a.code AS address, o.amount,
                       COALESCE(st.code, 'UNKNOWN') AS script_type
                FROM tx_output o
                LEFT JOIN address a ON a.id = o.fk_address_id
                LEFT JOIN script_type st ON st.id = o.fk_script_type_id
                WHERE o.fk_transaction_id = ANY(%s)
                ORDER BY o.fk_transaction_id, o.n
                """,
                (transaction_ids,),
            )
            outputs = pd.DataFrame(
                cursor.fetchall(),
                columns=["tx_id", "output_index", "address", "amount", "script_type"],
            ).dropna(subset=["address"])

            # CoinJoin viene calcolato sulla stessa finestra temporale, prima
            # del clustering, per poter bloccare le fusioni sospette.
            cursor.execute(
                """
                SELECT t.id, t.n_inputs,
                       COALESCE(ARRAY_AGG(o.amount ORDER BY o.n)
                                FILTER (WHERE o.id IS NOT NULL),
                                ARRAY[]::numeric[])
                FROM transaction t
                LEFT JOIN tx_output o ON o.fk_transaction_id = t.id
                WHERE t.id = ANY(%s)
                GROUP BY t.id, t.n_inputs
                ORDER BY t.id
                """,
                (transaction_ids,),
            )
            coinjoin_module = load_module(
                "coinjoin_detector", WALLET_ROOT / "coinjoin_detection" / "algorithm.py"
            )
            coinjoin_rows = []
            for transaction_id, n_inputs, output_amounts in cursor.fetchall():
                result = coinjoin_module.classify_transaction(n_inputs, output_amounts or [])
                coinjoin_rows.append({
                    "transaction_id": transaction_id,
                    "coinjoin_score": result["score"],
                    "candidate": result["score"] >= 0.75,
                })

    return inputs, outputs, pd.DataFrame(coinjoin_rows), transaction_ids
