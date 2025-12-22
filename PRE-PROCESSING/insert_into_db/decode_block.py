import os
import json
import traceback
from decimal import Decimal
import psycopg2
from dotenv import load_dotenv, find_dotenv

# =====================
# ENV & DB
# =====================
dotenv_path = find_dotenv()
load_dotenv(dotenv_path=".env")

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

BLOCKS_DIR = "./blocks"
LOG_FILE = "import_errors.log"

# =====================
# LOG ERROR
# =====================
def log_error(block_height, error):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write("\n============================\n")
        log.write(f"ERRORE BLOCCO {block_height}\n")
        log.write(str(error) + "\n")
        log.write(traceback.format_exc())
        log.write("============================\n")

# =====================
# MAIN LOOP
# =====================
for height in range(0, 10):

    try:
        filepath = os.path.join(BLOCKS_DIR, f"block_{height}.json")
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            block = json.load(f, parse_float=Decimal)

        # -----------------
        # BLOCK
        # -----------------
        cur.execute(
            "INSERT INTO block (id, hash, time) VALUES (%s,%s,%s)",
            (height, block["hash"], block["time"])
        )

        # -----------------
        # TRANSACTIONS
        # -----------------
        for tx in block["tx"]:
            txid = tx["txid"]

            cur.execute(
                "INSERT INTO transaction (txid, fk_block_id) VALUES (%s,%s) RETURNING id",
                (txid, height)
            )
            tx_db_id = cur.fetchone()[0]

            # =================
            # VIN
            # =================
            for vin in tx["vin"]:
                if "coinbase" in vin:
                    # coinbase input
                    cur.execute(
                        """
                        INSERT INTO tx_input
                        (fk_transaction_id, prev_transaction_id, prev_vout, pubkey_hex)
                        VALUES (%s, NULL, NULL, NULL)
                        """,
                        (tx_db_id,)
                    )
                    continue

                prev_txid = vin["txid"]
                prev_vout = vin["vout"]

                cur.execute(
                    "SELECT id FROM transaction WHERE txid=%s",
                    (prev_txid,)
                )
                res = cur.fetchone()
                prev_tx_db_id = res[0] if res else None

                # estrazione pubkey da scriptSig
                pubkey_hex = None
                asm = vin.get("scriptSig", {}).get("asm", "")
                if asm:
                    parts = asm.split(" ")
                    if len(parts) >= 2:
                        pubkey_hex = parts[-1]

                cur.execute(
                    """
                    INSERT INTO tx_input
                    (fk_transaction_id, prev_transaction_id, prev_vout, pubkey_hex)
                    VALUES (%s,%s,%s,%s)
                    """,
                    (tx_db_id, prev_tx_db_id, prev_vout, pubkey_hex)
                )

            # =================
            # VOUT
            # =================
            for vout in tx["vout"]:
                spk = vout["scriptPubKey"]

                script_type = spk["type"]
                script_hex = spk["hex"]
                amount = vout["value"]
                n = vout["n"]

                pubkey_hex = None
                fk_address_id = None

                # P2PK (coinbase early)
                if script_type == "pubkey":
                    # <len><pubkey><OP_CHECKSIG>
                    pubkey_hex = script_hex[2:-2]

                # P2PKH / P2SH / ecc.
                elif "addresses" in spk:
                    address = spk["addresses"][0]
                    cur.execute(
                        """
                        INSERT INTO address (code)
                        VALUES (%s)
                        ON CONFLICT DO NOTHING
                        RETURNING id
                        """,
                        (address,)
                    )
                    res = cur.fetchone()
                    if not res:
                        cur.execute("SELECT id FROM address WHERE code=%s", (address,))
                        res = cur.fetchone()
                    fk_address_id = res[0]

                cur.execute(
                    """
                    INSERT INTO tx_output
                    (fk_transaction_id, n, amount,
                     script_type, script_hex, pubkey_hex, fk_address_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (tx_db_id, n, amount,
                     script_type, script_hex, pubkey_hex, fk_address_id)
                )

        conn.commit()
        print(f"[OK] Blocco {height} importato")

    except Exception as e:
        conn.rollback()
        log_error(height, e)
        print(f"[ERRORE] Blocco {height}")

print("IMPORT COMPLETATO")