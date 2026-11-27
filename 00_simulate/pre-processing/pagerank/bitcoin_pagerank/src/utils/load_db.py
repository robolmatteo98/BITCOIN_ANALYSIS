"""Costruzione degli archi PageRank direttamente dal database PostgreSQL."""

from __future__ import annotations

import csv
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


# Permette al modulo PageRank di riutilizzare la configurazione DB del progetto
# senza duplicare la lettura del file .env.
REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "00_simulate").is_dir()
)
SIMULATE_ROOT = REPO_ROOT / "00_simulate"
if str(SIMULATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATE_ROOT))

from wallet_clustering.db import database_parameters  # noqa: E402


SATOSHIS_PER_BTC = Decimal("100000000")


def load_wallet_mapping(path: str | Path) -> dict[str, str]:
    """Carica address -> wallet_id dal report del wallet clustering."""
    mapping: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as mapping_file:
        for row in csv.DictReader(mapping_file):
            address = row.get("address")
            wallet_id = row.get("wallet_id")
            if address and wallet_id:
                # Il prefisso evita di confondere un wallet_id con un address
                # quando i risultati di modalità diverse vengono confrontati.
                mapping[address] = f"wallet_{wallet_id}"
    return mapping


def _to_satoshis(amount: Decimal) -> int:
    """Converte il DECIMAL BTC del DB nel valore intero usato da PageRank."""
    return int((Decimal(str(amount)) * SATOSHIS_PER_BTC).to_integral_value(
        rounding=ROUND_HALF_UP
    ))


def load_db_edges(
    database: str | None = None,
    transaction_limit: int | None = None,
    transaction_offset: int = 0,
    wallet_mapping: str | Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Costruisce gli archi address->address o wallet->wallet dal DB.

    Ogni input viene collegato a ogni output della stessa transazione, come nel
    decoder .dat precedente. Se viene fornito un mapping, gli indirizzi non
    presenti nel mapping vengono esclusi e gli archi interni allo stesso
    wallet vengono rimossi.
    """
    import psycopg2

    address_to_wallet = (
        load_wallet_mapping(wallet_mapping) if wallet_mapping is not None else None
    )
    limit_clause = "LIMIT %s" if transaction_limit is not None else ""
    offset_clause = "OFFSET %s" if transaction_offset else ""
    query = f"""
        WITH selected_transactions AS (
            SELECT id
            FROM transaction
            ORDER BY id
            {limit_clause}
            {offset_clause}
        )
        SELECT t.id AS tx_id,
               b.time AS timestamp,
               src.code AS src_address,
               dst.code AS dst_address,
               o.amount
        FROM tx_input i
        JOIN selected_transactions selected
          ON selected.id = i.fk_transaction_id
        JOIN transaction t ON t.id = i.fk_transaction_id
        JOIN block b ON b.id = t.fk_block_id
        JOIN tx_output previous_output
          ON previous_output.fk_transaction_id = i.prev_transaction_id
         AND previous_output.n = i.prev_vout
        JOIN address src ON src.id = previous_output.fk_address_id
        JOIN tx_output o ON o.fk_transaction_id = t.id
        JOIN address dst ON dst.id = o.fk_address_id
        ORDER BY t.id, i.id, o.n
    """
    params = ()
    if transaction_limit is not None:
        params += (transaction_limit,)
    if transaction_offset:
        params += (transaction_offset,)

    edges: list[dict[str, object]] = []
    skipped_unmapped = 0
    skipped_self_loops = 0
    with psycopg2.connect(**database_parameters(database)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            for _, timestamp, src_address, dst_address, amount in cursor.fetchall():
                src = str(src_address)
                dst = str(dst_address)
                if address_to_wallet is not None:
                    src = address_to_wallet.get(src)
                    dst = address_to_wallet.get(dst)
                    if src is None or dst is None:
                        skipped_unmapped += 1
                        continue
                if src == dst:
                    skipped_self_loops += 1
                    continue
                edges.append({
                    "src": src,
                    "dst": dst,
                    "value_sat": _to_satoshis(amount),
                    "timestamp": int(timestamp) if timestamp is not None else None,
                })

    return edges, {
        "skipped_unmapped": skipped_unmapped,
        "skipped_self_loops": skipped_self_loops,
    }
