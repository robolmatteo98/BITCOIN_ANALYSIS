import os
import json
import traceback
from decimal import Decimal
import psycopg2
from how_to_decode_address import pubkey_to_address
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path=".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER} to DB {DB_NAME}")

print(dotenv_path)

# Connessione al DB
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cur = conn.cursor()

BLOCKS_DIR = "./DB/blocks"
LOG_FILE = "import_errors.log"

def log_error(block_height, error):
    """Scrive informazioni dettagliate sull'errore in un file di log."""
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write("\n============================\n")
        log.write(f"ERRORE BLOCCO {block_height}\n")
        log.write(f"Tipo errore: {type(error).__name__}\n")
        log.write(f"Messaggio: {error}\n")
        log.write("Stacktrace:\n")
        log.write(traceback.format_exc())
        log.write("============================\n")

for height in range(0, 10000):

    try:
        filename = f"block_{height}.json"
        filepath = os.path.join(BLOCKS_DIR, filename)

        if not os.path.exists(filepath):
            print(f"[SKIP] Blocco {height} non trovato")
            continue

        with open(filepath, 'r', encoding="utf-8") as file:
            block = json.load(file, parse_float=Decimal)

        hash_block = block['hash']
        time = block['time']
        nTx = block['nTx']

        # INSERT BLOCK
        cur.execute(
            """
            INSERT INTO block (id, hash, time)
            VALUES (%s, %s, %s)
            """,
            (height, hash_block, time)
        )

        # LOOP TRANSAZIONI
        for tx in block['tx']:
            txid = tx['txid']

            cur.execute(
                """
                INSERT INTO transaction (txid, fk_block_id)
                VALUES (%s, %s)
                RETURNING id
                """,
                (txid, height)
            )
            result = cur.fetchone()
            transaction_id = result[0] if result else None

            # VIN
            tx_ins = []
            for vin_item in tx['vin']:
                if 'coinbase' in vin_item:
                    cur.execute(
                    "INSERT INTO tx_input (fk_transaction_id) VALUES (%s)",
                        (transaction_id,)
                    )
                    continue

                prev_txid = vin_item['txid']
                prev_vout = vin_item['vout']

                cur.execute(
                    "SELECT id FROM transaction WHERE txid = %s",
                    (prev_txid,)
                )
                result = cur.fetchone()
                prev_transaction_id = result[0] if result else None

                tx_ins.append((transaction_id, prev_transaction_id, prev_vout))

            if tx_ins:
                cur.executemany(
                    """
                    INSERT INTO tx_input
                    (fk_transaction_id, prev_transaction_id, prev_vout)
                    VALUES (%s, %s, %s)
                    """,
                    tx_ins
                )

            # VOUT
            tx_outs = []
            for vout_item in tx['vout']:
                address = pubkey_to_address(vout_item["scriptPubKey"])
                amount = vout_item["value"]
                index = vout_item["n"]

                tx_outs.append((transaction_id, index, amount, address))

                # se è una coinbase allora lo inserisce come indirizzo provvisorio, altrimenti come indirizzo reale?
                cur.execute(
                    """
                        INSERT INTO address (code)
                        VALUES (%s)
                        ON CONFLICT DO NOTHING
                    """,
                    (address,)
                )

            if tx_outs:
                cur.executemany(
                    """
                    INSERT INTO tx_output
                    (fk_transaction_id, n, amount, fk_address_code)
                    VALUES (%s, %s, %s, %s)
                    """,
                    tx_outs
                )

        conn.commit()
        print(f"[OK] Blocco {height} importato!")

    except Exception as e:
        conn.rollback()  # evita transazioni parziali
        print(f"[ERRORE] Blocco {height} NON importato: {e}")
        log_error(height, e)
        # continua al prossimo blocco

print("FINITO: Importazione completata")