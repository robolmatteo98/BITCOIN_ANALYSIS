#!/usr/bin/env python3
"""Decodifica i file blk*.dat e importa i blocchi in PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import find_dotenv, load_dotenv

from blocks_decoder import MAINNET_MAGIC, parse_block

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def database_parameters(database: str | None) -> dict[str, object]:
    load_dotenv(find_dotenv() or PROJECT_ROOT / ".env")
    required = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"variabili mancanti nel .env: {', '.join(missing)}")
    selected_database = database or os.getenv("DB_NAME")
    if not selected_database:
        raise RuntimeError("specificare --database oppure configurare DB_NAME nel .env")
    return {
        "dbname": selected_database,
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
    }


def iter_dat_blocks(path: Path, prevout_index: dict):
    payload = path.read_bytes()
    offset = 0
    while offset + 8 <= len(payload):
        block_start = offset
        magic = payload[offset:offset + 4]
        size = int.from_bytes(payload[offset + 4:offset + 8], "little")
        offset += 8
        if magic != MAINNET_MAGIC:
            raise ValueError(f"magic bytes non validi in {path} a offset {block_start}")
        block_bytes = payload[offset:offset + size]
        offset += size
        if len(block_bytes) != size:
            raise ValueError(f"blocco incompleto in {path} a offset {block_start}")
        yield parse_block(block_bytes, path.name, block_start, size, prevout_index)


def import_block(cur, block: dict, height: int, txids: dict[str, int]) -> int:
    cur.execute(
        "INSERT INTO block (id, hash, time) VALUES (%s, %s, %s)",
        (height, block["hash"], block["time"]),
    )

    for tx in block["tx"]:
        txid = tx["txid"]
        outputs = tx["vout"]
        total_sat = sum(output["value_sat"] for output in outputs)
        cur.execute(
            """
            INSERT INTO transaction (txid, fk_block_id, n_inputs, n_outputs, total_amount)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (txid, height, len(tx["vin"]), len(outputs), total_sat),
        )
        transaction_id = cur.fetchone()[0]
        txids[txid] = transaction_id

        for vin in tx["vin"]:
            if not vin:
                cur.execute(
                    "INSERT INTO tx_input (fk_transaction_id) VALUES (%s)",
                    (transaction_id,),
                )
                continue
            cur.execute(
                """
                INSERT INTO tx_input (fk_transaction_id, prev_transaction_id, prev_vout)
                VALUES (%s, %s, %s)
                """,
                (transaction_id, txids.get(vin["txid"]), vin["vout"]),
            )

        for output in outputs:
            address = output["scriptPubKey"].get("address")
            address_id = None
            if address is not None:
                # Risolvi prima l'id dell'address, poi inserisci il tx_output.
                # In questo modo l'output non può riferirsi a un address
                # inesistente.
                cur.execute(
                    """
                    INSERT INTO address (code) VALUES (%s)
                    ON CONFLICT (code) DO UPDATE SET code = EXCLUDED.code
                    RETURNING id
                    """,
                    (address,),
                )
                address_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO tx_output
                    (fk_transaction_id, n, amount, fk_address_id)
                VALUES (%s, %s, %s, %s)
                """,
                (transaction_id, output["n"], output["value_btc"], address_id),
            )
    return len(block["tx"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat-dir", type=Path, required=True, help="directory containing blk*.dat")
    parser.add_argument("--database", metavar="NOME_DB", help="database PostgreSQL (default: DB_NAME)")
    parser.add_argument("--start-height", type=int, default=0, help="altezza del primo blocco (default: 0)")
    parser.add_argument("--limit-blocks", type=int, help="importa solo i primi N blocchi")
    args = parser.parse_args()

    dat_files = sorted(args.dat_dir.glob("blk*.dat"))
    if not dat_files:
        print(f"Nessun file blk*.dat trovato in {args.dat_dir}", file=sys.stderr)
        return 2
    if args.start_height < 0 or (args.limit_blocks is not None and args.limit_blocks < 1):
        parser.error("--start-height deve essere >= 0 e --limit-blocks > 0")

    imported_blocks = 0
    imported_transactions = 0
    prevout_index: dict = {}
    txids: dict[str, int] = {}

    try:
        with psycopg2.connect(**database_parameters(args.database)) as connection:
            with connection.cursor() as cur:
                cur.execute("SELECT txid, id FROM transaction")
                txids.update(dict(cur.fetchall()))
                for dat_file in dat_files:
                    for block in iter_dat_blocks(dat_file, prevout_index):
                        if args.limit_blocks is not None and imported_blocks >= args.limit_blocks:
                            break
                        imported_transactions += import_block(
                            cur, block, args.start_height + imported_blocks, txids
                        )
                        imported_blocks += 1
                        if imported_blocks % 100 == 0:
                            connection.commit()
                            print(f"Blocchi importati: {imported_blocks}")
                    if args.limit_blocks is not None and imported_blocks >= args.limit_blocks:
                        break
                connection.commit()
    except Exception as exc:
        print(f"ERRORE: importazione interrotta: {exc}", file=sys.stderr)
        return 2

    print(f"Importazione completata: {imported_blocks} blocchi, {imported_transactions} transazioni")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
