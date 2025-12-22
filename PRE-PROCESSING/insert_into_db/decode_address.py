import hashlib

BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def base58_encode(b: bytes) -> str:
    n = int.from_bytes(b, 'big')
    res = ''
    while n > 0:
        n, r = divmod(n, 58)
        res = BASE58_ALPHABET[r] + res
    pad = len(b) - len(b.lstrip(b'\x00'))
    return '1' * pad + res

def pubkey_to_p2pkh(pubkey_hex: str, mainnet=True) -> str:
    pubkey = bytes.fromhex(pubkey_hex)
    sha = hashlib.sha256(pubkey).digest()
    ripe = hashlib.new('ripemd160', sha).digest()
    prefix = b'\x00' if mainnet else b'\x6f'
    payload = prefix + ripe
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58_encode(payload + checksum)

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(".env")

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

print("Avvio risoluzione indirizzi P2PK...")

# ==========================
# 1️⃣ Trova UTXO P2PK spesi
# ==========================
cur.execute("""
    SELECT
        o.id AS output_id,
        o.pubkey_hex,
        i.pubkey_hex AS input_pubkey
    FROM tx_output o
    JOIN tx_input i
      ON i.prev_transaction_id = o.fk_transaction_id
     AND i.prev_vout = o.n
    WHERE
        o.script_type = 'pubkey'
        AND o.fk_address_id IS NULL
        AND i.pubkey_hex IS NOT NULL
        AND o.pubkey_hex = i.pubkey_hex
""")

rows = cur.fetchall()
print(f"Trovati {len(rows)} output risolvibili")

# ==========================
# 2️⃣ Risolvi indirizzi
# ==========================
for output_id, pubkey_hex, _ in rows:

    address = pubkey_to_p2pkh(pubkey_hex)

    # inserisci indirizzo (se non esiste)
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

    address_id = res[0]

    # aggiorna output
    cur.execute(
        """
        UPDATE tx_output
        SET fk_address_id = %s, spent = TRUE
        WHERE id = %s
        """,
        (address_id, output_id)
    )

conn.commit()
print("Risoluzione completata")