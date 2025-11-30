import psycopg2
import datetime
from collections import Counter
from dotenv import load_dotenv
import os

# Carica variabili dal .env
load_dotenv(dotenv_path=".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER} to DB {DB_NAME}")

# Connessione al DB
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

## Recupera tutte le regioni dal DB
cursor.execute("SELECT id, utc_start, utc_end FROM region")
regions = cursor.fetchall()

def guess_region_id(utc_offset):
    for r in regions:
        rid, start, end = r
        if start <= utc_offset <= end:
            return rid
    return None


# Recupera tutti gli indirizzi
cursor.execute("SELECT address FROM count_tx")
addresses = [row[0] for row in cursor.fetchall()]

for address in addresses:
    # Prendiamo tutte le transazioni di quell'indirizzo
    cursor.execute("""
        SELECT b.time
        FROM bitcoin_tx_output o
        JOIN bitcoin_transaction t ON t.id = o.fk_transaction_id
        JOIN bitcoin_block b ON b.id = t.fk_block_id
        WHERE o.fk_address_code = %s
    """, (address,))
    timestamps = [row[0] for row in cursor.fetchall()]

    n_tx = len(timestamps)

    # Ore UTC delle transazioni
    hours = [datetime.datetime.utcfromtimestamp(ts).hour for ts in timestamps]
    hour_counts = Counter(hours)
    hour_activity = sorted(hour_counts.items())

    # Lista ripetuta di ore
    active_hours = [h for h, c in hour_activity for _ in range(c)]

    # Testiamo tutti i fusi orari (offset da -12 a +12)
    best_offset = 0
    best_activity_score = 0
    active_range = list(range(10, 23))  # ore diurne 10-22

    for offset in range(-12, 13):
        shifted_hours = [(h + offset) % 24 for h in active_hours]
        score = sum(1 for h in shifted_hours if h in active_range)
        if score > best_activity_score:
            best_activity_score = score
            best_offset = offset

    # Stima area geografica
    region = guess_region_id(best_offset)

    print(f"Address: {address} | Tx: {n_tx} | UTC offset: {best_offset:+} | Region: {region}")

    # Aggiorna la tabella bitcoin_address
    cursor.execute("""
        UPDATE bitcoin_address
        SET region_id = %s
        WHERE code = %s
    """, (region, address))
    conn.commit()

conn.close()